//! The peer test uses authenticated SSH tunnels; HTTP never leaves loopback.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{io::Read, net::SocketAddr, path::Path};
use subtle::ConstantTimeEq;

pub fn loopback(address: SocketAddr) -> Result<()> {
    ensure!(
        address.ip().is_loopback() && address.port() != 0,
        "Qwen requires a nonzero loopback endpoint; use the SSH tunnel workflow in docs/secure-peer-test.md"
    );
    Ok(())
}

pub fn next_hop(address: SocketAddr, allowed: &[SocketAddr]) -> Result<()> {
    loopback(address)?;
    ensure!(
        allowed.contains(&address),
        "downstream endpoint is not in --allow-next"
    );
    Ok(())
}

pub fn token_matches(expected: &str, provided: &str) -> bool {
    let expected = Sha256::digest(expected.as_bytes());
    let provided = Sha256::digest(provided.as_bytes());
    bool::from(expected.ct_eq(&provided))
}

pub fn read_token(path: &Path) -> Result<String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .context("open token file (symlinks are not allowed)")?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.len() <= 257,
        "invalid token file size/type"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            meta.permissions().mode() & 0o077 == 0,
            "token file must be private: chmod 600 FILE"
        );
    }
    let mut value = String::new();
    file.take(258).read_to_string(&mut value)?;
    let value = value.trim_end_matches(['\n', '\r']).to_owned();
    crate::server::validate_token(&value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_public_wildcard_private_and_unapproved_destinations() {
        for address in [
            "0.0.0.0:7901",
            "192.168.1.2:7901",
            "100.64.0.1:7901",
            "8.8.8.8:7901",
            "127.0.0.1:0",
            "[::]:7901",
        ] {
            assert!(loopback(address.parse().unwrap()).is_err());
        }
        let peer = "127.0.0.1:7902".parse().unwrap();
        assert!(next_hop(peer, &[peer]).is_ok());
        assert!(next_hop(peer, &[]).is_err());
        assert!(next_hop("127.0.0.1:22".parse().unwrap(), &[peer]).is_err());
        assert!(loopback("[::1]:7901".parse().unwrap()).is_ok());
    }
    #[test]
    fn compares_token_digests() {
        assert!(token_matches("secret", "secret"));
        assert!(!token_matches("secret", "Secret"));
        assert!(!token_matches("secret", ""));
    }
    #[cfg(unix)]
    #[test]
    fn rejects_readable_or_symlinked_secret_files() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let dir = std::env::temp_dir().join(format!("p2p-token-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("token");
        std::fs::write(&path, "0123456789abcdef0123456789abcdef\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_token(&path).is_err());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_token(&path).is_ok());
        let link = dir.join("link");
        symlink(&path, &link).unwrap();
        assert!(read_token(&link).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
