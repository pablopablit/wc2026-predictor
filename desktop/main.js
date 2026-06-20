// Electron main process: spawn the local Python API, then open the window.
const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const PY = path.join(ROOT, ".venv", "bin", "python");
const PORT = 8765;

let server = null;
let win = null;

function startServer() {
  server = spawn(PY, ["-m", "wc2026.serve", "--port", String(PORT)], { cwd: ROOT });
  server.stdout.on("data", (d) => console.log("[server]", d.toString().trim()));
  server.stderr.on("data", (d) => console.log("[server]", d.toString().trim()));
  server.on("exit", (code) => console.log("[server] exited", code));
}

function createWindow() {
  win = new BrowserWindow({
    width: 940,
    height: 700,
    minWidth: 720,
    minHeight: 560,
    title: "WC2026 Predictor",
    backgroundColor: "#ffffff",
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  win.loadFile(path.join(__dirname, "index.html"));
}

function stopServer() {
  if (server) {
    server.kill();
    server = null;
  }
}

app.whenReady().then(() => {
  startServer();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  stopServer();
  if (process.platform !== "darwin") app.quit();
  else app.quit();
});

app.on("before-quit", stopServer);
