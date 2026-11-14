package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"rag-hybrid/internal/config"
	"rag-hybrid/internal/engine"
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
  index <path> [--recreate]     index .md/.txt/.rst files (recursive)
  retrieve <query>              hybrid retrieval + rerank (no LLM)
  ask <query>                   full RAG answer via DeepSeek`)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

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
	fmt.Printf("rag-hybrid serving on http://%s\n", addr)
	srv := &http.Server{
		Addr:              addr,
		Handler:           server.New(cfg).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	if err := srv.ListenAndServe(); err != nil {
		fatalf("server: %v", err)
	}
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
	result, err := engine.NewIndexer(cfg).IndexFiles(ctx, positional[0], *recreate)
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
	hits, err := engine.NewSearcher(cfg).Retrieve(ctx, question)
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
	answer, err := engine.NewSearcher(cfg).Ask(ctx, question)
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
