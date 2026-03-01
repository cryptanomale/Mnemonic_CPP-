#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include "cuda_sha512.cuh"

// ---------------------------------------------------------------------------
// HMAC-SHA512 on CUDA
// ---------------------------------------------------------------------------

__device__ void cuda_hmac_sha512(
    const uint8_t* __restrict__ key,   uint32_t key_len,
    const uint8_t* __restrict__ msg,   uint32_t msg_len,
    uint8_t* __restrict__ out)          // 64 bytes
{
    // Keys longer than 128 bytes must be hashed first
    uint8_t k[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) k[i] = 0;

    if (key_len > 128) {
        cuda_sha512(key, key_len, k);
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < key_len; i++) k[i] = key[i];
    }

    // ipad / opad
    uint8_t ipad[128], opad[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    // inner = SHA512(ipad || msg)
    // max inner input: 128 + 256 = 384 bytes => 3 blocks, but for our use
    // msg is at most ~200 bytes so 2 blocks suffice
    uint8_t inner_data[384];
    #pragma unroll
    for (int i = 0; i < 128; i++) inner_data[i] = ipad[i];
    for (uint32_t i = 0; i < msg_len; i++) inner_data[128 + i] = msg[i];
    uint32_t inner_len = 128 + msg_len;

    uint8_t inner_hash[64];
    // Choose 1 or 2 block variant
    if (inner_len <= 128)
        cuda_sha512(inner_data, inner_len, inner_hash);
    else
        cuda_sha512_2block(inner_data, inner_len, inner_hash);

    // outer = SHA512(opad || inner_hash)
    uint8_t outer_data[192];  // 128 + 64
    #pragma unroll
    for (int i = 0; i < 128; i++) outer_data[i] = opad[i];
    #pragma unroll
    for (int i = 0; i < 64;  i++) outer_data[128 + i] = inner_hash[i];

    cuda_sha512_2block(outer_data, 192, out);
}

// ---------------------------------------------------------------------------
// PBKDF2-HMAC-SHA512
// BIP39: password = mnemonic, salt = "mnemonic" + passphrase
//        iterations = 2048, dk_len = 64
// ---------------------------------------------------------------------------

__device__ void cuda_pbkdf2_hmac_sha512(
    const uint8_t* __restrict__ password, uint32_t pass_len,
    const uint8_t* __restrict__ salt,     uint32_t salt_len,
    uint8_t* __restrict__ dk)             // 64 bytes output
{
    // U1 = HMAC(password, salt || INT(1))
    uint8_t salt_block[256];
    #pragma unroll
    for (uint32_t i = 0; i < salt_len; i++) salt_block[i] = salt[i];
    // Append block index = 1 as 4-byte big-endian
    salt_block[salt_len + 0] = 0x00;
    salt_block[salt_len + 1] = 0x00;
    salt_block[salt_len + 2] = 0x00;
    salt_block[salt_len + 3] = 0x01;

    uint8_t U[64], T[64];
    cuda_hmac_sha512(password, pass_len, salt_block, salt_len + 4, U);

    // T = U1
    #pragma unroll
    for (int i = 0; i < 64; i++) T[i] = U[i];

    // Iterations 2..2048: U_n = HMAC(password, U_{n-1})
    for (int iter = 1; iter < 2048; iter++) {
        uint8_t tmp[64];
        cuda_hmac_sha512(password, pass_len, U, 64, tmp);
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            U[i]  = tmp[i];
            T[i] ^= tmp[i];
        }
    }

    #pragma unroll
    for (int i = 0; i < 64; i++) dk[i] = T[i];
}
