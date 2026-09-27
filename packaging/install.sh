#!/bin/sh
# User-level binary installation. Never asks for sudo or downloads model weights.
set -eu
version=${SANGAMA_VERSION:-v0.1.0-preview.1}
case "$version" in v[0-9]* ) ;; *) echo 'Invalid release version' >&2; exit 1;; esac
case "$version" in *[!A-Za-z0-9._-]*) echo 'Invalid release version' >&2; exit 1;; esac
os=$(uname -s); arch=$(uname -m)
case "$os/$arch" in
 Darwin/arm64) target=aarch64-apple-darwin ;;
 Linux/x86_64) target=x86_64-unknown-linux-gnu ;;
 Linux/aarch64|Linux/arm64) target=aarch64-unknown-linux-gnu ;;
 *) echo "Unsupported platform: $os/$arch. Windows users: use install.ps1." >&2; exit 1 ;;
esac
bin=${SANGAMA_BIN_DIR:-"$HOME/.local/bin"}
case "$bin" in /*) ;; *) echo 'Install directory must be absolute' >&2; exit 1;; esac
umask 077
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
asset="sangama-$version-$target.tar.gz"
base="https://github.com/devdil/sangama/releases/download/$version"
fetch() {
 if [ -n "${SANGAMA_RELEASE_DIR:-}" ]; then
  cp "$SANGAMA_RELEASE_DIR/$1" "$scratch/$1"
 else
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --tlsv1.2 "$base/$1" -o "$scratch/$1"
 fi
}
fetch "$asset"
fetch SHA256SUMS
expected=$(awk -v name="$asset" '$2 == name {print $1}' "$scratch/SHA256SUMS")
[ ${#expected} -eq 64 ] || { echo 'Missing or ambiguous checksum' >&2; exit 1; }
case "$expected" in *[!0-9a-f]*) echo 'Invalid checksum' >&2; exit 1;; esac
if command -v sha256sum >/dev/null 2>&1; then
 actual=$(sha256sum "$scratch/$asset" | awk '{print $1}')
else
 actual=$(shasum -a 256 "$scratch/$asset" | awk '{print $1}')
fi
[ "$actual" = "$expected" ] || { echo 'Checksum mismatch; nothing installed' >&2; exit 1; }
# Extract only the fixed executable member, never arbitrary archive paths.
tar -xzf "$scratch/$asset" -C "$scratch" sangama
[ -f "$scratch/sangama" ] && [ ! -L "$scratch/sangama" ] || { echo 'Invalid executable' >&2; exit 1; }
mkdir -p "$bin"
install -m 755 "$scratch/sangama" "$bin/.sangama-new-$$"
mv -f "$bin/.sangama-new-$$" "$bin/sangama"
printf 'Installed %s to %s/sangama\n' "$version" "$bin"
"$bin/sangama" --version
printf 'Run: "%s/sangama" doctor\nWorker setup: https://github.com/devdil/sangama/blob/main/docs/distribution.md\n' "$bin"
printf 'Add this directory to PATH if needed. Your shell profile was not modified.\n'
