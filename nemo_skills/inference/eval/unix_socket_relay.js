#!/usr/bin/env node
"use strict";

const net = require("net");
const fs = require("fs");

function parseArgs(argv) {
  const args = {};
  for (let index = 2; index < argv.length; index += 2) {
    args[argv[index]] = argv[index + 1];
  }
  return args;
}

const args = parseArgs(process.argv);
const port = Number(args["--port"]);
const socketPath = args["--socket"];
const readyFile = args["--ready-file"];

if (!Number.isInteger(port) || !socketPath) {
  throw new Error("usage: unix_socket_relay.js --port PORT --socket PATH");
}

const server = net.createServer((client) => {
  const upstream = net.createConnection(socketPath);
  client.pipe(upstream);
  upstream.pipe(client);
  const close = () => {
    client.destroy();
    upstream.destroy();
  };
  client.on("error", close);
  upstream.on("error", close);
});

server.listen(port, "127.0.0.1", () => {
  if (readyFile) {
    fs.writeFileSync(readyFile, "");
  }
});
