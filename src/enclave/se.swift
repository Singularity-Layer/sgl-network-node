// Secure Enclave identity key for sgl-node (macOS only). Compiled by build.rs into a static
// library and linked into the node, so the node stays ONE binary (one allowlisted hash).
//
// The private key is generated INSIDE the Secure Enclave and never leaves it. CryptoKit hands
// back `dataRepresentation`, an enclave-encrypted blob that only THIS Mac's enclave can use, so
// the node can keep it in a plain file without any keychain entitlement (verified on a real Mac,
// unsigned binary, 2026-10-30). Copying the file to another machine yields nothing usable.
//
// What this proves: the identity is bound to one physical Mac. What it does NOT prove: that the
// software asking the enclave to sign is untouched (the enclave signs whatever it is given).
//
// C ABI (all buffers caller-owned). Return 0 on success, negative on error:
//   -1 enclave unavailable, -2 key create/load failed, -3 buffer too small, -4 sign failed.

import CryptoKit
import Foundation

private let pubLen = 65   // ANSI X9.63 uncompressed: 0x04 || X(32) || Y(32)
private let sigLen = 64   // raw r || s (IEEE P1363), what WebCrypto ECDSA verify expects

@_cdecl("sgl_se_available")
public func sgl_se_available() -> Int32 {
    return SecureEnclave.isAvailable ? 1 : 0
}

@_cdecl("sgl_se_create")
public func sgl_se_create(
    _ outBlob: UnsafeMutablePointer<UInt8>,
    _ blobCap: Int,
    _ outBlobLen: UnsafeMutablePointer<Int>,
    _ outPub: UnsafeMutablePointer<UInt8>
) -> Int32 {
    guard SecureEnclave.isAvailable else { return -1 }
    guard let key = try? SecureEnclave.P256.Signing.PrivateKey() else { return -2 }
    let blob = key.dataRepresentation
    guard blob.count <= blobCap else { return -3 }
    blob.copyBytes(to: outBlob, count: blob.count)
    outBlobLen.pointee = blob.count
    let pub = key.publicKey.x963Representation
    guard pub.count == pubLen else { return -2 }
    pub.copyBytes(to: outPub, count: pubLen)
    return 0
}

private func load(_ blob: UnsafePointer<UInt8>, _ blobLen: Int) -> SecureEnclave.P256.Signing.PrivateKey? {
    guard SecureEnclave.isAvailable else { return nil }
    let data = Data(bytes: blob, count: blobLen)
    return try? SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: data)
}

@_cdecl("sgl_se_public_key")
public func sgl_se_public_key(
    _ blob: UnsafePointer<UInt8>,
    _ blobLen: Int,
    _ outPub: UnsafeMutablePointer<UInt8>
) -> Int32 {
    guard SecureEnclave.isAvailable else { return -1 }
    guard let key = load(blob, blobLen) else { return -2 }
    let pub = key.publicKey.x963Representation
    guard pub.count == pubLen else { return -2 }
    pub.copyBytes(to: outPub, count: pubLen)
    return 0
}

@_cdecl("sgl_se_sign")
public func sgl_se_sign(
    _ blob: UnsafePointer<UInt8>,
    _ blobLen: Int,
    _ msg: UnsafePointer<UInt8>,
    _ msgLen: Int,
    _ outSig: UnsafeMutablePointer<UInt8>
) -> Int32 {
    guard SecureEnclave.isAvailable else { return -1 }
    guard let key = load(blob, blobLen) else { return -2 }
    let message = Data(bytes: msg, count: msgLen)
    guard let sig = try? key.signature(for: message) else { return -4 }
    let raw = sig.rawRepresentation
    guard raw.count == sigLen else { return -4 }
    raw.copyBytes(to: outSig, count: sigLen)
    return 0
}
