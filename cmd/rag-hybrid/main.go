package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"rag-hybrid/internal/config"
	"rag-hybrid/internal/deepseek"
	"rag-hybrid/internal/engine"
	"rag-hybrid/internal/openrouter"
	"rag-hybrid/internal/qdrant"
	"rag-hybrid/internal/server"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		cmdServe(os.Args[2:])
	case "index":
		cmdIndex(os.Args[2:])
	case "retrieve":
		cmdRetrieve(os.Args[2:])
	case "ask":
		cmdAsk(os.Args[2:])
	case "forget":
		cmdForget(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

// splitFlags separates dash-prefixed flags from positional arguments so
// both "index --recreate path" and "index path --recreate" work.
func splitFlags(args []string) (flagArgs, positional []string) {
	for _, arg := range args {
		if strings.HasPrefix(arg, "-") {
			flagArgs = append(flagArgs, arg)
		} else {
			positional = append(positional, arg)
		}
	}
	return flagArgs, positional
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: rag-hybrid <command> [args]

commands:
  serve [--host H] [--port P]   run the web UI (defaults from WEB_HOST/WEB_PORT)
  index <path> [--recreate]     index .md/.txt/.rst/.pdf files (recursive)
  retrieve <query>              hybrid retrieval + rerank (no LLM)
  ask <query>                   full RAG answer via DeepSeek
  forget <source>               remove every chunk indexed from a file`)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

// cmdForget removes every chunk indexed from one source file. The
// argument must match the indexed payload source exactly (as listed by
// GET /api/documents or the UI document list).
func cmdForget(args []string) {
	source := strings.TrimSpace(strings.Join(args, " "))
	if source == "" {
		fatalf("usage: rag-hybrid forget <source>")
	}
	cfg, err := config.Load()
	if err != nil {
		fatalf("config: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	vs := qdrant.NewClient(cfg.QdrantURL, cfg.QdrantAPIKey, cfg.QdrantCollection)
	if err := vs.DeleteBySource(ctx, source); err != nil {
		fatalf("forget: %v", err)
	}
	fmt.Printf("Forgot %q: every chunk indexed from it is gone\n", source)
}

// newIndexer is the CLI composition root for indexing: it builds the
// concrete driven adapters and injects them into the engine.
func newIndexer(cfg *config.Config) *engine.Indexer {
	vectorStore := qdrant.NewClient(cfg.QdrantURL, cfg.QdrantAPIKey, cfg.QdrantCollection)
	return engine.NewIndexer(cfg, vectorStore, openrouter.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL))
}

// newSearcher is the CLI composition root for the query side.
func newSearcher(cfg *config.Config) *engine.Searcher {
	vectorStore := qdrant.NewClient(cfg.QdrantURL, cfg.QdrantAPIKey, cfg.QdrantCollection)
	open := openrouter.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL)
	return engine.NewSearcher(cfg, vectorStore, open, open, deepseek.New(cfg.DeepSeekAPIKey, cfg.LLMBaseURL))
}

// drainTimeout bounds how long shutdown waits for in-flight requests (SSE
// generations included) before force-closing their connections. Keep it
// below the docker stop grace period (compose sets stop_grace_period).
var drainTimeout = 10 * time.Second

func cmdServe(args []string) {
	flags := flag.NewFlagSet("serve", flag.ExitOnError)
	host := flags.String("host", "", "listen host (default WEB_HOST)")
	port := flags.Int("port", 0, "listen port (default WEB_PORT)")
	_ = flags.Parse(args)

	cfg, err := config.Load()
	if err != nil {
		fatalf("config: %v", err)
	}
	if *host != "" {
		cfg.WebHost = *host
	}
	if *port != 0 {
		cfg.WebPort = *port
	}
	addr := fmt.Sprintf("%s:%d", cfg.WebHost, cfg.WebPort)
	// Bind before opening the store: a busy port fails fast, with nothing
	// to clean up.
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fatalf("listen: %v", err)
	}
	fmt.Printf("rag-hybrid serving on http://%s\n", ln.Addr())

	// SIGINT (Ctrl+C) and SIGTERM (docker stop) trigger the graceful drain.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := runServer(ctx, cfg, ln); err != nil {
		fatalf("%v", err)
	}
}

// runServer serves the web UI until ctx is cancelled or the listener
// fails. On cancellation it stops accepting connections, waits up to
// drainTimeout for in-flight requests to finish, force-closes whatever
// is still open past the deadline, and only then releases the app
// resources (retention loop, SQLite store). Serve returns ErrServerClosed
// as soon as the listeners close, BEFORE draining finishes, so the
// <-drained wait is what actually orders store.Close after the last
// handler. app.Close is called exactly once on either path.
func runServer(ctx context.Context, cfg *config.Config, ln net.Listener) error {
	app, err := server.New(cfg)
	if err != nil {
		return fmt.Errorf("server: %w", err)
	}
	srv := &http.Server{
		Handler:           app.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	drained := make(chan struct{})
	go func() {
		defer close(drained)
		<-ctx.Done()
		fmt.Println("shutdown: signal received, draining in-flight requests")
		drainCtx, cancel := context.WithTimeout(context.Background(), drainTimeout)
		defer cancel()
		if err := srv.Shutdown(drainCtx); err != nil {
			fmt.Printf("shutdown: drain deadline exceeded, forcing close: %v\n", err)
			_ = srv.Close()
		}
	}()

	err = srv.Serve(ln)
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		app.Close() // never served: release the store before failing
		return fmt.Errorf("server: %w", err)
	}
	<-drained
	app.Close()
	fmt.Println("shutdown: complete (connections drained, store closed)")
	return nil
}

func cmdIndex(args []string) {
	flags := flag.NewFlagSet("index", flag.ExitOnError)
	recreate := flags.Bool("recreate", false, "drop and recreate the collection first")
	flagArgs, positional := splitFlags(args)
	_ = flags.Parse(flagArgs)
	if len(positional) != 1 {
		fatalf("usage: rag-hybrid index <path> [--recreate]")
	}
	cfg, err := config.Load()
	if err != nil {
		fatalf("config: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	result, err := newIndexer(cfg).IndexFiles(ctx, positional[0], *recreate)
	if err != nil {
		fatalf("index: %v", err)
	}
	fmt.Printf("Indexed %d file(s) -> %d chunk(s) in collection '%s'\n", result.Files, result.Chunks, cfg.QdrantCollection)
}

func cmdRetrieve(args []string) {
	question := strings.Join(args, " ")
	if strings.TrimSpace(question) == "" {
		fatalf("usage: rag-hybrid retrieve <query>")
	}
	cfg, err := config.Load()
	if err != nil {
		fatalf("config: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	hits, err := newSearcher(cfg).Retrieve(ctx, question)
	if err != nil {
		fatalf("retrieve: %v", err)
	}
	if len(hits) == 0 {
		fmt.Println("No documents found. Did you index anything? (rag-hybrid index <path>)")
		os.Exit(1)
	}
	for _, hit := range hits {
		preview := hit.Content
		if len(preview) > 140 {
			preview = preview[:140]
		}
		fmt.Printf("[%.3f] %s :: %s\n", hit.Score, hit.Source, preview)
	}
}

func cmdAsk(args []string) {
	question := strings.Join(args, " ")
	if strings.TrimSpace(question) == "" {
		fatalf("usage: rag-hybrid ask <query>")
	}
	cfg, err := config.Load()
	if err != nil {
		fatalf("config: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	answer, err := newSearcher(cfg).Ask(ctx, question)
	if err != nil {
		fatalf("ask: %v", err)
	}
	fmt.Println(answer.Text)
	if len(answer.Sources) > 0 {
		fmt.Println("\nSources:")
		for _, hit := range answer.Sources {
			fmt.Printf("  - %s\n", hit.Source)
		}
	}
}
