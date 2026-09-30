//! Apply address policy at the transport boundary, including addresses learned
//! through Kademlia. Filtering only signed application advertisements is insufficient.
use libp2p::{
    Multiaddr, Transport,
    core::transport::{DialOpts, ListenerId, TransportError, TransportEvent},
    multiaddr::Protocol,
    quic, tcp,
};
use std::{
    net::SocketAddr,
    pin::Pin,
    task::{Context, Poll},
};
/// A transport that only dials the pinned relay or, unless relaying is forced, public
/// addresses. The same policy applies to TCP and QUIC.
pub struct Guarded<T> {
    inner: T,
    pinned_relay: Option<Endpoint>,
    force_relay: bool,
}
pub type GuardedTcp = Guarded<tcp::tokio::Transport>;
pub type GuardedQuic = Guarded<quic::tokio::Transport>;
impl GuardedTcp {
    pub fn new(relay: Option<&Multiaddr>, force_relay: bool) -> Self {
        Self {
            inner: tcp::tokio::Transport::new(tcp::Config::default().nodelay(true)),
            pinned_relay: relay.and_then(endpoint),
            force_relay,
        }
    }
}
impl GuardedQuic {
    pub fn new(
        key: &libp2p::identity::Keypair,
        relay: Option<&Multiaddr>,
        force_relay: bool,
    ) -> Self {
        Self {
            inner: quic::tokio::Transport::new(quic::Config::new(key)),
            pinned_relay: relay.and_then(endpoint),
            force_relay,
        }
    }
}
impl<T> Guarded<T> {
    fn allowed(&self, address: &Multiaddr) -> bool {
        let Some(endpoint) = endpoint(address) else {
            return false;
        };
        if Some(endpoint) == self.pinned_relay {
            return true;
        }
        !self.force_relay && public(address)
    }
}
/// An IPv4 socket and whether it is QUIC over UDP rather than TCP.
type Endpoint = (SocketAddr, bool);
fn endpoint(address: &Multiaddr) -> Option<Endpoint> {
    let mut parts = address.iter();
    match (parts.next(), parts.next(), parts.next()) {
        (Some(Protocol::Ip4(ip)), Some(Protocol::Tcp(port)), _) if port > 0 => {
            Some((SocketAddr::new(ip.into(), port), false))
        }
        (Some(Protocol::Ip4(ip)), Some(Protocol::Udp(port)), Some(Protocol::QuicV1))
            if port > 0 =>
        {
            Some((SocketAddr::new(ip.into(), port), true))
        }
        _ => None,
    }
}
fn socket(address: &Multiaddr) -> Option<SocketAddr> {
    endpoint(address).map(|(socket, _)| socket)
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
impl<T: Transport + Unpin> Transport for Guarded<T> {
    type Output = T::Output;
    type Error = T::Error;
    type ListenerUpgrade = T::ListenerUpgrade;
    type Dial = T::Dial;
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
        // QUIC follows the same policy, and a TCP pin does not admit the same port over UDP.
        assert!(g.allowed(&"/ip4/1.1.1.1/udp/9000/quic-v1".parse().unwrap()));
        assert!(!g.allowed(&"/ip4/10.1.1.1/udp/9000/quic-v1".parse().unwrap()));
        assert!(!g.allowed(&"/ip4/1.1.1.1/udp/9000".parse().unwrap()));
        let quic_relay: Multiaddr = "/ip4/10.1.1.1/udp/9000/quic-v1".parse().unwrap();
        let q = GuardedTcp::new(Some(&quic_relay), true);
        assert!(q.allowed(&quic_relay) && !q.allowed(&relay));
    }
}
