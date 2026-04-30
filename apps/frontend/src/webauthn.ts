/** WebAuthn / passkey helpers — browser ↔ backend bridge.
 *
 * Backend ships base64url-encoded bytes; the browser's WebAuthn API needs
 * ArrayBuffers in specific spots (challenge, user.id, allowCredentials[].id).
 */
import { api } from "./api";

function b64urlToBytes(s: string): Uint8Array {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  const b64 = (s + pad).replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function bytesToB64url(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function decodePublicKeyCreate(pk: any): PublicKeyCredentialCreationOptions {
  return {
    ...pk,
    challenge: b64urlToBytes(pk.challenge),
    user: { ...pk.user, id: b64urlToBytes(pk.user.id) },
    excludeCredentials: (pk.excludeCredentials || []).map((c: any) => ({
      ...c,
      id: b64urlToBytes(c.id),
    })),
  };
}

function decodePublicKeyGet(pk: any): PublicKeyCredentialRequestOptions {
  return {
    ...pk,
    challenge: b64urlToBytes(pk.challenge),
    allowCredentials: (pk.allowCredentials || []).map((c: any) => ({
      ...c,
      id: b64urlToBytes(c.id),
    })),
  };
}

export function isWebAuthnSupported(): boolean {
  return typeof window !== "undefined" && !!window.PublicKeyCredential && !!navigator.credentials;
}

/** Register a new passkey for the currently-authenticated user. */
export async function registerPasskey(name: string): Promise<{ id: number; name: string }> {
  if (!isWebAuthnSupported()) throw new Error("This browser does not support passkeys");

  const begin = await api.post<{ challenge_id: string; publicKey: any }>(
    "/api/webauthn/register/begin",
    {},
  );
  const options = decodePublicKeyCreate(begin.publicKey);

  const cred = (await navigator.credentials.create({ publicKey: options })) as PublicKeyCredential | null;
  if (!cred) throw new Error("Registration cancelled");

  const att = cred.response as AuthenticatorAttestationResponse;
  const transports = (att as any).getTransports ? (att as any).getTransports() : [];

  return await api.post("/api/webauthn/register/complete", {
    challenge_id: begin.challenge_id,
    name,
    id: cred.id,
    rawId: bytesToB64url(cred.rawId),
    type: cred.type,
    response: {
      clientDataJSON: bytesToB64url(att.clientDataJSON),
      attestationObject: bytesToB64url(att.attestationObject),
      transports,
    },
  });
}

/** Sign in with a passkey. Returns access + refresh JWTs. */
export async function loginWithPasskey(username?: string): Promise<{ access_token: string; refresh_token: string; must_change_password?: boolean }> {
  if (!isWebAuthnSupported()) throw new Error("This browser does not support passkeys");

  const begin = await api.post<{ challenge_id: string; publicKey: any }>(
    "/api/webauthn/login/begin",
    username ? { username } : {},
  );
  const options = decodePublicKeyGet(begin.publicKey);

  const cred = (await navigator.credentials.get({ publicKey: options })) as PublicKeyCredential | null;
  if (!cred) throw new Error("Sign-in cancelled");

  const ass = cred.response as AuthenticatorAssertionResponse;

  return await api.post("/api/webauthn/login/complete", {
    challenge_id: begin.challenge_id,
    id: cred.id,
    rawId: bytesToB64url(cred.rawId),
    type: cred.type,
    response: {
      clientDataJSON: bytesToB64url(ass.clientDataJSON),
      authenticatorData: bytesToB64url(ass.authenticatorData),
      signature: bytesToB64url(ass.signature),
      userHandle: ass.userHandle ? bytesToB64url(ass.userHandle) : null,
    },
  });
}
