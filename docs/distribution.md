# Worker downloads and installers

Sangama's worker can be distributed without Rust or Python installed on the peer.
The native `mesh --config` command manages its own shard worker child process.
These packages install/launch that worker; they do not yet replace network onboarding
with a fully automatic setup wizard.

## Package formats

| Platform | Package | Backend |
|---|---|---|
| Apple Silicon macOS | `.dmg` with Sangama.app; `.tar.gz` command-line binary | Metal or CPU |
| Linux x86-64 | `.tar.gz` command-line binary | CPU |
| Linux ARM64 | `.tar.gz` command-line binary | CPU |
| Windows x64 | `.zip` containing `sangama.exe` and `start-worker.cmd` | CPU |

Linux binaries use glibc and require a compatible distribution (x86-64 built on Ubuntu 22.04; ARM64 on Ubuntu 24.04). Alpine/musl, Intel Macs and Windows ARM64 are not release targets yet. Windows GPU acceleration is not included. Windows packages may need the Microsoft Visual C++ runtime supplied by the operating environment.

## Install from a published release

The commands below become usable only after the specified release and its assets
are published in an accessible repository. The source repository is currently private, so anonymous downloads are not available even for a published release there. A public download destination must be configured before advertising curl setup. CI artifacts and draft releases are not anonymous download endpoints.
Do not advertise these commands as live before publication.

macOS/Linux (preview version example):

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/devdil/sangama/releases/download/v0.1.0-preview.1/install.sh | sh
```

The installer selects a native archive, verifies its SHA-256 against the release's
`SHA256SUMS`, and installs to `~/.local/bin`. It does not request sudo, edit your
shell profile, start a service or download model weights. Use an absolute
`SANGAMA_BIN_DIR` to override the location. For another release, set
`SANGAMA_VERSION` on the `sh` process too. An unsupported platform or checksum
mismatch stops installation.

For Windows, download and inspect `install.ps1` from the release, then run it in
PowerShell under your normal account, subject to your organization's script policy:

```powershell
.\install.ps1 -Version v0.1.0-preview.1
& "$env:LOCALAPPDATA\Sangama\bin\sangama.exe" doctor
```

Alternatively, extract the Windows ZIP and run `sangama.exe` directly. Do not disable
system execution or security policy to bypass a blocked download. The installer
does not elevate privileges or edit PATH. No network invitation belongs in shell
history or installer URLs.

A checksum fetched from the same release protects against corruption, not a
compromised release account/server. The curl script itself executes before binary
verification: inspect it or download it separately if needed. Public distribution
should use signed/notarized macOS packages and Authenticode-signed Windows binaries.
The current packaging supports a supplied macOS signing identity; without one it
produces an ad-hoc-signed development DMG. That is not Developer ID signing or
notarization. Unsigned Windows builds may trigger SmartScreen. Do not instruct
peers to bypass OS security warnings; complete signing before broad distribution.

## Start an invited worker

You still need the operator's network configuration and pinned authority public key,
a private invitation file, and the prepared supported model shard files. Keep private
state under your own user profile. On Windows, use a private directory under
`%LOCALAPPDATA%` with access limited to your account and trusted system administrators;
do not store identities or tokens in a shared directory.

```sh
sangama mesh-identity --state-dir /absolute/private/state
sangama mesh-join --config /absolute/path/worker.json \
  --invitation-file /absolute/private/invitation
sangama mesh --config /absolute/path/worker.json
```

Use `sangama.exe` and Windows paths in PowerShell. Use managed mode in the JSON
configuration, as shown in [the mesh guide](admitted-mesh.md). This avoids the older
Python process launcher. Each peer still needs the consistent bridge map and model
files. Allocation/loading is controlled through the local UI or `mesh-allocate`.

On macOS, drag Sangama.app from the DMG to Applications and open it. The app asks
for your existing worker JSON configuration and opens Terminal to run the worker.
On Windows, `start-worker.cmd` asks for that configuration path. These launchers
are not a graphical onboarding wizard. Stop a foreground worker with Ctrl-C.

Uninstall by removing the installed executable/app. Model files, identity, tokens
and membership are separate: stop the worker and revoke membership first if leaving
a network. Installers deliberately do not delete those files or silently rotate keys.

## Build and validate packages

```sh
./scripts/cargo build --release --locked --features metal
python3 packaging/build.py --version v0.1.0-preview.1 \
  --target aarch64-apple-darwin --binary target/release/sangama --dmg
python3 -m unittest discover -s tests -p 'test_packaging.py'
```

Other platforms omit `--features metal` and use their native target name; Windows
passes `target/release/sangama.exe`. The output directory defaults to ignored `dist/`.
Use a clean output directory for each release. `SANGAMA_CODESIGN_IDENTITY` can name
an installed Developer ID certificate for macOS signing. Notarization/stapling must
be completed and DMG checksums regenerated before public release; no developer
certificate is embedded in this repository.

The **Worker packages** GitHub Actions workflow builds/tests all four native targets
and uploads packages as CI artifacts. A manual dispatch can optionally aggregate
assets into a **draft prerelease** for review. It does not publish it automatically.
Signing secrets, real macOS/Windows installation acceptance, automatic shard
downloads, resumable downloads, service installation, automatic updates and
one-click invitation handoff remain separate release gates.

Windows available-memory detection follows Microsoft's
[MEMORYSTATUSEX definition](https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex).
The build matrix uses [GitHub-hosted native runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
