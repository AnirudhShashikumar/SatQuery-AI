#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SatQuery AI — macOS Installer  (install_satquery_app.sh)
#
# Creates a native SatQuery AI.app that launches the platform
# with a single double-click from Finder / Dock.
#
# Usage:
#   cd "<project-root>/sar-colorization-app"
#   bash scripts/macos/install_satquery_app.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Resolve paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
MACOS_SCRIPTS="${PROJECT_ROOT}/scripts/macos"

echo "╔════════════════════════════════════════════════════════╗"
echo "║     SatQuery AI — macOS Application Installer          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo ""

# ─── 1. Validate Python environment ──────────────────────────
echo "▸ Checking Python environment…"
PYTHON="${PROJECT_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "✗ Python venv not found at: ${PYTHON}"
    echo ""
    echo "  Create it with:"
    echo "    cd \"${PROJECT_ROOT}\""
    echo "    python3 -m venv venv"
    echo "    venv/bin/pip install -r requirements-backend.txt"
    exit 1
fi
PYTHON_VER=$("${PYTHON}" --version 2>&1)
echo "  ✓ ${PYTHON_VER}"

# ─── 2. Validate Node/npm ────────────────────────────────────
echo "▸ Checking Node.js…"
if ! command -v node &>/dev/null; then
    echo "✗ Node.js not found."
    echo "  Install from https://nodejs.org/ or: brew install node"
    exit 1
fi
if ! command -v npm &>/dev/null; then
    echo "✗ npm not found."
    exit 1
fi
echo "  ✓ Node $(node --version), npm $(npm --version)"

# ─── Build identity ──────────────────────────────────────────
GIT_SHA=$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo "no-git")
BUILD_TIMESTAMP=$(date -u "+%Y%m%dT%H%M%SZ")
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null || true)" ]]; then
    SOURCE_STATE="dirty"
else
    SOURCE_STATE="clean"
fi
SATQUERY_BUILD_ID="${GIT_SHA}-${BUILD_TIMESTAMP}-${SOURCE_STATE}"
BUILD_ID_FILE="${FRONTEND_DIR}/.satquery-build-id"
GIT_SHA_FILE="${FRONTEND_DIR}/.satquery-git-sha"
echo "  Build identity: ${SATQUERY_BUILD_ID}"

# ─── 3. Validate frontend dependencies ───────────────────────
echo "▸ Checking frontend dependencies…"
if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    echo "  Installing npm packages…"
    (cd "${FRONTEND_DIR}" && npm ci --prefer-offline 2>&1 | tail -1)
fi
echo "  ✓ node_modules present"

# ─── 4. Create .env.local ────────────────────────────────────
echo "▸ Configuring local API URL…"
ENV_LOCAL="${FRONTEND_DIR}/.env.local"
if [[ ! -f "${ENV_LOCAL}" ]] || ! grep -q "NEXT_PUBLIC_API_URL" "${ENV_LOCAL}" 2>/dev/null; then
    cat > "${ENV_LOCAL}" <<'EOF'
# SatQuery AI — Local launcher configuration.
NEXT_PUBLIC_API_URL=http://127.0.0.1:8010
EOF
fi
echo "  ✓ NEXT_PUBLIC_API_URL=http://127.0.0.1:8010"

# ─── 5. Build frontend (production) ──────────────────────────
echo "▸ Building frontend for production…"
echo "  (This may take 30–60 seconds on first run)"
(cd "${FRONTEND_DIR}" && \
    NEXT_PUBLIC_API_URL=http://127.0.0.1:8010 \
    NEXT_PUBLIC_SATQUERY_BUILD_ID="${SATQUERY_BUILD_ID}" \
    NEXT_PUBLIC_SATQUERY_GIT_SHA="${GIT_SHA}" \
    npm run build 2>&1 | tail -5)
if [[ ! -d "${FRONTEND_DIR}/.next" ]]; then
    echo "✗ Frontend build failed."
    exit 1
fi
if ! grep -R -F -q "${SATQUERY_BUILD_ID}" "${FRONTEND_DIR}/.next" 2>/dev/null; then
    echo "✗ Frontend build identity was not embedded in the production output."
    exit 1
fi
printf '%s\n' "${SATQUERY_BUILD_ID}" > "${BUILD_ID_FILE}"
printf '%s\n' "${GIT_SHA}" > "${GIT_SHA_FILE}"
echo "  ✓ Production build ready"

# ─── 6. Generate macOS icon ──────────────────────────────────
echo "▸ Creating application icon…"

ICON_SOURCE="${FRONTEND_DIR}/public/icon-512.png"
ICONSET_DIR="${MACOS_SCRIPTS}/SatQueryAI.iconset"
ICNS_FILE="${MACOS_SCRIPTS}/SatQueryAI.icns"

if [[ -f "${ICON_SOURCE}" ]]; then
    mkdir -p "${ICONSET_DIR}"

    # Generate all required sizes from the 512x512 source
    sips -z   16   16 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_16x16.png"    >/dev/null 2>&1
    sips -z   32   32 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_16x16@2x.png" >/dev/null 2>&1
    sips -z   32   32 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_32x32.png"    >/dev/null 2>&1
    sips -z   64   64 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_32x32@2x.png" >/dev/null 2>&1
    sips -z  128  128 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_128x128.png"    >/dev/null 2>&1
    sips -z  256  256 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_128x128@2x.png" >/dev/null 2>&1
    sips -z  256  256 "${ICON_SOURCE}" --out "${ICONSET_DIR}/icon_256x256.png"    >/dev/null 2>&1
    cp "${ICON_SOURCE}"                       "${ICONSET_DIR}/icon_256x256@2x.png"
    cp "${ICON_SOURCE}"                       "${ICONSET_DIR}/icon_512x512.png"
    # For 512@2x we need 1024, but source is only 512 — use 512 as best available
    cp "${ICON_SOURCE}"                       "${ICONSET_DIR}/icon_512x512@2x.png"

    iconutil -c icns "${ICONSET_DIR}" -o "${ICNS_FILE}" 2>/dev/null || true
    rm -rf "${ICONSET_DIR}"

    if [[ -f "${ICNS_FILE}" ]]; then
        echo "  ✓ Icon generated from project branding"
    else
        echo "  ⚠ iconutil failed — app will use default icon"
    fi
else
    echo "  ⚠ No icon source found — app will use default icon"
fi

# ─── 7. Make launcher scripts executable ─────────────────────
chmod +x "${MACOS_SCRIPTS}/start_satquery.sh"
chmod +x "${MACOS_SCRIPTS}/stop_satquery.sh"

# ─── 8. Create the .app bundle (native Swift binary) ──────────
echo "▸ Creating SatQuery AI.app…"

APP_NAME="SatQuery AI"
INSTALL_DIR="${HOME}/Applications"
INSTALLED_APP="${INSTALL_DIR}/${APP_NAME}.app"
APP_MACOS="${INSTALLED_APP}/Contents/MacOS"
APP_RESOURCES="${INSTALLED_APP}/Contents/Resources"

# Remove any previous installation
rm -rf "${INSTALLED_APP}"
rm -rf "${MACOS_SCRIPTS}/${APP_NAME}.app"
mkdir -p "${APP_MACOS}" "${APP_RESOURCES}"

# Write the Swift launcher source in a disposable build directory so installer
# reruns never leave generated source in the repository.
SWIFT_BUILD_BASE="${TMPDIR:-/private/tmp}"
[[ -d "${SWIFT_BUILD_BASE}" ]] || SWIFT_BUILD_BASE="/private/tmp"
SWIFT_BUILD_DIR=$(mktemp -d "${SWIFT_BUILD_BASE%/}/SatQueryAI.XXXXXX")
SWIFT_SRC="${SWIFT_BUILD_DIR}/SatQueryLauncher.swift"
trap 'rm -rf "${SWIFT_BUILD_DIR}"' EXIT
cat > "${SWIFT_SRC}" <<'SWIFTEOF'
import Foundation
@main
struct SatQueryLauncher {
    static func main() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let configPath = "\(home)/Library/Application Support/SatQueryAI/config"
        let logDir = "\(home)/Library/Logs/SatQueryAI"
        guard let configContents = try? String(contentsOfFile: configPath, encoding: .utf8) else {
            showError("SatQuery AI project not found.\n\nRe-run the installer:\n  bash scripts/macos/install_satquery_app.sh")
            return
        }
        var projectRoot = ""
        for line in configContents.components(separatedBy: "\n") {
            if line.hasPrefix("SATQUERY_PROJECT_ROOT=") {
                projectRoot = line
                    .replacingOccurrences(of: "SATQUERY_PROJECT_ROOT=", with: "")
                    .replacingOccurrences(of: "\"", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }
        guard !projectRoot.isEmpty else {
            showError("SatQuery AI project path is empty.\n\nRe-run the installer.")
            return
        }
        let launcherScript = "\(projectRoot)/scripts/macos/start_satquery.sh"
        guard FileManager.default.fileExists(atPath: launcherScript) else {
            showError("Launcher not found at:\n\(launcherScript)\n\nRe-run the installer.")
            return
        }
        try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        let logFile = "\(logDir)/launcher.log"
        let path = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        let existingPath = ProcessInfo.processInfo.environment["PATH"] ?? ""
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [launcherScript]
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "\(path):\(existingPath)"
        process.environment = env
        if let handle = FileHandle(forWritingAtPath: logFile) {
            handle.seekToEndOfFile(); process.standardOutput = handle; process.standardError = handle
        } else {
            FileManager.default.createFile(atPath: logFile, contents: nil)
            if let handle = FileHandle(forWritingAtPath: logFile) {
                process.standardOutput = handle; process.standardError = handle
            }
        }
        do {
            try process.run(); process.waitUntilExit()
            if process.terminationStatus != 0 {
                showError("SatQuery AI startup failed (exit \(process.terminationStatus)).\n\nSee: ~/Library/Logs/SatQueryAI/launcher.log")
            }
        } catch {
            showError("Failed to launch SatQuery AI:\n\n\(error.localizedDescription)")
        }
    }
    static func showError(_ message: String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        p.arguments = ["-e", "display dialog \"\(message.replacingOccurrences(of: "\"", with: "\\\""))\" buttons {\"OK\"} default button \"OK\" with icon stop with title \"SatQuery AI\""]
        try? p.run(); p.waitUntilExit()
    }
}
SWIFTEOF

# Compile to native Mach-O binary
echo "  Compiling native launcher…"
if ! swiftc -o "${APP_MACOS}/SatQueryAI" "${SWIFT_SRC}" -parse-as-library 2>&1; then
    echo "✗ Swift compilation failed. Xcode Command Line Tools required."
    echo "  Install with: xcode-select --install"
    exit 1
fi
rm -rf "${SWIFT_BUILD_DIR}"
trap - EXIT

# Create Info.plist
cat > "${INSTALLED_APP}/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>SatQuery AI</string>
    <key>CFBundleDisplayName</key>
    <string>SatQuery AI</string>
    <key>CFBundleIdentifier</key>
    <string>com.satquery.launcher</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>SatQueryAI</string>
    <key>CFBundleIconFile</key>
    <string>SatQueryAI</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# Copy icon
if [[ -f "${ICNS_FILE}" ]]; then
    cp "${ICNS_FILE}" "${APP_RESOURCES}/SatQueryAI.icns"
fi

# Remove quarantine so double-click works immediately
xattr -dr com.apple.quarantine "${INSTALLED_APP}" 2>/dev/null || true
touch "${INSTALLED_APP}"

echo "  ✓ Installed to: ${INSTALLED_APP}"

# ─── 9. Store project root configuration ─────────────────────
echo "▸ Storing configuration…"
STATE_DIR="${HOME}/Library/Application Support/SatQueryAI"
mkdir -p "${STATE_DIR}"

cat > "${STATE_DIR}/config" <<CONFIGEOF
# SatQuery AI — Project configuration
# Generated by install_satquery_app.sh on $(date)
# Re-run the installer if the project moves.
SATQUERY_PROJECT_ROOT="${PROJECT_ROOT}"
SATQUERY_FRONTEND_BUILD_ID="${SATQUERY_BUILD_ID}"
SATQUERY_FRONTEND_GIT_SHA="${GIT_SHA}"
CONFIGEOF

echo "  ✓ Config saved to: ${STATE_DIR}/config"

# ─── Done ─────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                   Installation Complete                ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║                                                        ║"
echo "║  App location:                                         ║"
echo "║    ~/Applications/SatQuery AI.app                      ║"
echo "║                                                        ║"
echo "║  To launch:                                            ║"
echo "║    • Double-click from Finder / Launchpad              ║"
echo "║    • Or: open ~/Applications/SatQuery\ AI.app          ║"
echo "║                                                        ║"
echo "║  To stop:                                              ║"
echo "║    bash \"${MACOS_SCRIPTS}/stop_satquery.sh\""
echo "║                                                        ║"
echo "║  Logs:  ~/Library/Logs/SatQueryAI/                     ║"
echo "║  State: ~/Library/Application Support/SatQueryAI/      ║"
echo "║                                                        ║"
echo "║  Backend:  http://127.0.0.1:8010                       ║"
echo "║  Frontend: http://127.0.0.1:3000                       ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Optionally launch immediately. Automation can set
# SATQUERY_INSTALL_LAUNCH=yes|no; Finder-oriented runs still prompt.
case "${SATQUERY_INSTALL_LAUNCH:-ask}" in
    yes|YES|true|TRUE|1)
        open "${INSTALLED_APP}"
        ;;
    no|NO|false|FALSE|0)
        ;;
    *)
        read -r -p "Launch SatQuery AI now? [Y/n] " response
        response="${response:-Y}"
        if [[ "${response}" =~ ^[Yy] ]]; then
            open "${INSTALLED_APP}"
        fi
        ;;
esac
