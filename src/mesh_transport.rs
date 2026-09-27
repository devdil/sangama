//! Apply address policy at the transport boundary, including addresses learned
//! through Kademlia. Filtering only signed application advertisements is insufficient.
use libp2p::{
    Multiaddr, Transport,
    core::transport::{DialOpts, ListenerId, TransportError, TransportEvent},
    multiaddr::Protocol,
    tcp,
};
use std::{
    net::SocketAddr,
    pin::Pin,
    task::{Context, Poll},
};
pub struct GuardedTcp {
    inner: tcp::tokio::Transport,
    pinned_relay: Option<SocketAddr>,
    force_relay: bool,
}
impl GuardedTcp {
    pub fn new(relay: Option<&Multiaddr>, force_relay: bool) -> Self {
        Self {
            inner: tcp::tokio::Transport::new(tcp::Config::default().nodelay(true)),
            pinned_relay: relay.and_then(socket),
            force_relay,
        }
    }
    fn allowed(&self, address: &Multiaddr) -> bool {
        let Some(socket) = socket(address) else {
            return false;
        };
        if Some(socket) == self.pinned_relay {
            return true;
        }
        !self.force_relay && public(address)
    }
}
fn socket(address: &Multiaddr) -> Option<SocketAddr> {
    let mut parts = address.iter();
    match (parts.next(), parts.next()) {
        (Some(Protocol::Ip4(ip)), Some(Protocol::Tcp(port))) if port > 0 => {
            Some(SocketAddr::new(ip.into(), port))
        }
        _ => None,
    }
}
pub fn public(address: &Multiaddr) -> bool {
    let Some(SocketAddr::V4(s)) = socket(address) else {
        return false;
    };
    let ip = s.ip();
    let octets = ip.octets();
    !ip.is_private()
        && !ip.is_loopback()
        && !ip.is_link_local()
        && !ip.is_unspecified()
        && !ip.is_multicast()
        && !ip.is_broadcast()
        && !ip.is_documentation()
        && octets[0] != 0
        && octets[0] < 224
        && !(octets[0] == 100 && (64..=127).contains(&octets[1]))
        && !(octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
}
impl Transport for GuardedTcp {
    type Output = <tcp::tokio::Transport as Transport>::Output;
    type Error = <tcp::tokio::Transport as Transport>::Error;
    type ListenerUpgrade = <tcp::tokio::Transport as Transport>::ListenerUpgrade;
    type Dial = <tcp::tokio::Transport as Transport>::Dial;
    fn listen_on(
        &mut self,
        id: ListenerId,
        a: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        self.inner.listen_on(id, a)
    }
    fn remove_listener(&mut self, id: ListenerId) -> bool {
        self.inner.remove_listener(id)
    }
    fn dial(
        &mut self,
        a: Multiaddr,
        o: DialOpts,
    ) -> Result<Self::Dial, TransportError<Self::Error>> {
        if !self.allowed(&a) {
            return Err(TransportError::MultiaddrNotSupported(a));
        }
        self.inner.dial(a, o)
    }
    fn poll(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        Pin::new(&mut self.get_mut().inner).poll(cx)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn private_addresses_require_exact_operator_pin() {
        let relay: Multiaddr = "/ip4/10.1.1.1/tcp/9000".parse().unwrap();
        let g = GuardedTcp::new(Some(&relay), false);
        assert!(g.allowed(&relay));
        for address in [
            "/ip4/10.1.1.1/tcp/22",
            "/ip4/127.0.0.1/tcp/80",
            "/ip4/169.254.169.254/tcp/80",
            "/ip4/100.64.0.1/tcp/80",
            "/dns4/localhost/tcp/80",
            "/ip4/192.0.2.1/tcp/80",
        ] {
            assert!(!g.allowed(&address.parse().unwrap()));
        }
        let public = "/ip4/1.1.1.1/tcp/9000".parse().unwrap();
        assert!(g.allowed(&public));
        assert!(!GuardedTcp::new(Some(&relay), true).allowed(&public));
    }
}
