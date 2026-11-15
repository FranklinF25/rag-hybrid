# ---- build stage: static Go binary ----
FROM golang:1.27-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/rag-hybrid ./cmd/rag-hybrid

# ---- runtime stage: scratch + CA certs (HTTPS to OpenRouter/DeepSeek) ----
FROM scratch
COPY --from=build /out/rag-hybrid /rag-hybrid
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
ENV DATA_DIR=/data
USER 1000:1000
EXPOSE 8000
ENTRYPOINT ["/rag-hybrid"]
CMD ["serve", "--host", "0.0.0.0"]
