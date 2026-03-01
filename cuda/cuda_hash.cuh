#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// SHA-256 on CUDA
// ---------------------------------------------------------------------------

__constant__ uint32_t c_sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define SHA256_ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SHA256_SHR(x,n)  ((x)>>(n))
#define SHA256_S0(x) (SHA256_ROTR(x,2)  ^ SHA256_ROTR(x,13) ^ SHA256_ROTR(x,22))
#define SHA256_S1(x) (SHA256_ROTR(x,6)  ^ SHA256_ROTR(x,11) ^ SHA256_ROTR(x,25))
#define SHA256_s0(x) (SHA256_ROTR(x,7)  ^ SHA256_ROTR(x,18) ^ SHA256_SHR(x,3))
#define SHA256_s1(x) (SHA256_ROTR(x,17) ^ SHA256_ROTR(x,19) ^ SHA256_SHR(x,10))
#define SHA256_CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define SHA256_MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

__device__ void cuda_sha256(
    const uint8_t* __restrict__ data,
    uint32_t len,
    uint8_t* __restrict__ out)  // 32 bytes
{
    uint32_t W[64];
    uint8_t  block[64];
    for (int i = 0; i < 64; i++) block[i] = (i < (int)len) ? data[i] : 0;
    if (len < 64) block[len] = 0x80;
    // length bits big-endian in last 8 bytes
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 4; i++)
        block[63-i] = (uint8_t)(bitlen >> (i*8));

    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        W[i]  = ((uint32_t)block[i*4+0]) << 24;
        W[i] |= ((uint32_t)block[i*4+1]) << 16;
        W[i] |= ((uint32_t)block[i*4+2]) <<  8;
        W[i] |= ((uint32_t)block[i*4+3]);
    }
    #pragma unroll 48
    for (int i = 16; i < 64; i++)
        W[i] = SHA256_s1(W[i-2]) + W[i-7] + SHA256_s0(W[i-15]) + W[i-16];

    uint32_t a=0x6a09e667, b=0xbb67ae85, c=0x3c6ef372, d=0xa54ff53a;
    uint32_t e=0x510e527f, f=0x9b05688c, g=0x1f83d9ab, h=0x5be0cd19;

    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + SHA256_S1(e) + SHA256_CH(e,f,g) + c_sha256_k[i] + W[i];
        uint32_t T2 = SHA256_S0(a) + SHA256_MAJ(a,b,c);
        h=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
    }

    uint32_t H[8] = {
        0x6a09e667+a, 0xbb67ae85+b, 0x3c6ef372+c, 0xa54ff53a+d,
        0x510e527f+e, 0x9b05688c+f, 0x1f83d9ab+g, 0x5be0cd19+h
    };

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (H[i]>>24)&0xff;
        out[i*4+1] = (H[i]>>16)&0xff;
        out[i*4+2] = (H[i]>> 8)&0xff;
        out[i*4+3] = (H[i]    )&0xff;
    }
}

// ---------------------------------------------------------------------------
// RIPEMD-160 on CUDA
// ---------------------------------------------------------------------------

#define RMD_ROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define RMD_F(x,y,z) ((x)^(y)^(z))
#define RMD_G(x,y,z) (((x)&(y))|(~(x)&(z)))
#define RMD_H(x,y,z) (((x)|(~(y)))^(z))
#define RMD_I(x,y,z) (((x)&(z))|((y)&(~(z))))
#define RMD_J(x,y,z) ((x)^((y)|(~(z))))

__constant__ uint32_t c_rmd_KL[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__constant__ uint32_t c_rmd_KR[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

__constant__ uint8_t c_rmd_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__constant__ uint8_t c_rmd_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__constant__ uint8_t c_rmd_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__constant__ uint8_t c_rmd_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

__device__ void cuda_ripemd160(
    const uint8_t* __restrict__ data,
    uint32_t len,
    uint8_t* __restrict__ out)  // 20 bytes
{
    uint32_t W[16];
    uint8_t  block[64];
    for (int i = 0; i < 64; i++) block[i] = (i < (int)len) ? data[i] : 0;
    if (len < 64) block[len] = 0x80;
    uint64_t bitlen = (uint64_t)len * 8;
    // length in little-endian at bytes 56-63
    for (int i = 0; i < 8; i++)
        block[56+i] = (uint8_t)(bitlen >> (i*8));

    // Load little-endian words
    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        W[i]  = (uint32_t)block[i*4+0];
        W[i] |= (uint32_t)block[i*4+1] <<  8;
        W[i] |= (uint32_t)block[i*4+2] << 16;
        W[i] |= (uint32_t)block[i*4+3] << 24;
    }

    uint32_t al=0x67452301, bl=0xEFCDAB89, cl=0x98BADCFE, dl=0x10325476, el=0xC3D2E1F0;
    uint32_t ar=0x67452301, br=0xEFCDAB89, cr=0x98BADCFE, dr=0x10325476, er=0xC3D2E1F0;

    #pragma unroll 80
    for (int i = 0; i < 80; i++) {
        int rnd = i / 16;
        uint32_t fl, fr;
        switch(rnd) {
            case 0: fl=RMD_F(bl,cl,dl); fr=RMD_J(br,cr,dr); break;
            case 1: fl=RMD_G(bl,cl,dl); fr=RMD_I(br,cr,dr); break;
            case 2: fl=RMD_H(bl,cl,dl); fr=RMD_H(br,cr,dr); break;
            case 3: fl=RMD_I(bl,cl,dl); fr=RMD_G(br,cr,dr); break;
            default: fl=RMD_J(bl,cl,dl); fr=RMD_F(br,cr,dr); break;
        }
        uint32_t T;
        T = RMD_ROTL(al + fl + W[c_rmd_RL[i]] + c_rmd_KL[rnd], c_rmd_SL[i]) + el;
        al=el; el=dl; dl=RMD_ROTL(cl,10); cl=bl; bl=T;
        T = RMD_ROTL(ar + fr + W[c_rmd_RR[i]] + c_rmd_KR[rnd], c_rmd_SR[i]) + er;
        ar=er; er=dr; dr=RMD_ROTL(cr,10); cr=br; br=T;
    }

    uint32_t T = 0xEFCDAB89 + cl + dr;
    uint32_t H0 = 0x98BADCFE + dl + er;
    uint32_t H1 = 0x10325476 + el + ar;
    uint32_t H2 = 0x67452301 + al + br;
    uint32_t H3 = 0xC3D2E1F0 + bl + cr;
    uint32_t H4 = T;
    (void)H0; // reuse variable
    H0 = 0x67452301 + cl + dr; // fix ordering
    // Correct computation
    uint32_t h0 = 0x67452301 + cl + dr;
    uint32_t h1 = 0xEFCDAB89 + dl + er;
    uint32_t h2 = 0x98BADCFE + el + ar;
    uint32_t h3 = 0x10325476 + al + br;
    uint32_t h4 = 0xC3D2E1F0 + bl + cr;
    (void)T; (void)H0; (void)H1; (void)H2; (void)H3; (void)H4;

    // Write little-endian
    auto wr = [&](uint32_t v, int off) {
        out[off+0]=(v)&0xff; out[off+1]=(v>>8)&0xff;
        out[off+2]=(v>>16)&0xff; out[off+3]=(v>>24)&0xff;
    };
    wr(h0,0); wr(h1,4); wr(h2,8); wr(h3,12); wr(h4,16);
}

// ---------------------------------------------------------------------------
// HASH160 = RIPEMD160(SHA256(data))
// ---------------------------------------------------------------------------
__device__ void cuda_hash160(
    const uint8_t* __restrict__ data,
    uint32_t len,
    uint8_t* __restrict__ out)  // 20 bytes
{
    uint8_t sha[32];
    cuda_sha256(data, len, sha);
    cuda_ripemd160(sha, 32, out);
}

// ---------------------------------------------------------------------------
// Keccak-256 (Ethereum) on CUDA
// ---------------------------------------------------------------------------

__constant__ uint64_t c_keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ void keccak_f1600(uint64_t st[25]) {
    #pragma unroll 24
    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5], tmp;
        // Theta
        C[0]=st[0]^st[5]^st[10]^st[15]^st[20];
        C[1]=st[1]^st[6]^st[11]^st[16]^st[21];
        C[2]=st[2]^st[7]^st[12]^st[17]^st[22];
        C[3]=st[3]^st[8]^st[13]^st[18]^st[23];
        C[4]=st[4]^st[9]^st[14]^st[19]^st[24];
        D[0]=C[4]^((C[1]<<1)|(C[1]>>63));
        D[1]=C[0]^((C[2]<<1)|(C[2]>>63));
        D[2]=C[1]^((C[3]<<1)|(C[3]>>63));
        D[3]=C[2]^((C[4]<<1)|(C[4]>>63));
        D[4]=C[3]^((C[0]<<1)|(C[0]>>63));
        for(int i=0;i<25;i++) st[i]^=D[i%5];
        // Rho & Pi
        uint64_t B[25];
        const int rho[25]={0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
        const int pi[25] ={0,10,20,5,15,16,1,11,21,6,7,17,2,12,22,23,8,18,3,13,24,9,19,4,14};
        for(int i=0;i<25;i++) {
            int r=rho[i];
            B[pi[i]] = r ? ((st[i]<<r)|(st[i]>>(64-r))) : st[i];
        }
        // Chi
        for(int i=0;i<25;i++)
            st[i] = B[i] ^ (~B[(i+5)%25 - (i+5)%25 + ((i/5)*5 + (i+1)%5)] & B[(i/5)*5 + (i+2)%5]);
        // Correct Chi:
        for(int y=0;y<5;y++) {
            uint64_t t[5];
            for(int x=0;x<5;x++) t[x]=B[y*5+x];
            for(int x=0;x<5;x++) st[y*5+x]=t[x]^(~t[(x+1)%5]&t[(x+2)%5]);
        }
        // Iota
        st[0] ^= c_keccak_rc[round];
        (void)tmp;
    }
}

__device__ void cuda_keccak256(
    const uint8_t* __restrict__ data,
    uint32_t len,
    uint8_t* __restrict__ out)  // 32 bytes
{
    uint64_t st[25];
    #pragma unroll 25
    for(int i=0;i<25;i++) st[i]=0;

    // Absorb: rate = 136 bytes for Keccak-256
    uint8_t block[136];
    for(int i=0;i<136;i++) block[i] = (i < (int)len) ? data[i] : 0;
    // Keccak padding: 0x01 ... 0x80 (NOT SHA3 0x06)
    block[len] ^= 0x01;
    block[135] ^= 0x80;

    // XOR into state (little-endian 64-bit lanes)
    for(int i=0;i<17;i++) {
        uint64_t lane=0;
        for(int j=0;j<8;j++) lane |= ((uint64_t)block[i*8+j]) << (j*8);
        st[i] ^= lane;
    }
    keccak_f1600(st);

    // Squeeze: first 32 bytes
    for(int i=0;i<4;i++)
        for(int j=0;j<8;j++)
            out[i*8+j] = (st[i] >> (j*8)) & 0xff;
}

// Ethereum address = last 20 bytes of Keccak256(pubkey_64_bytes)
__device__ void cuda_eth_address(
    const uint8_t* __restrict__ pubkey64,  // uncompressed, no 0x04 prefix
    uint8_t* __restrict__ addr)             // 20 bytes
{
    uint8_t hash[32];
    cuda_keccak256(pubkey64, 64, hash);
    for(int i=0;i<20;i++) addr[i] = hash[12+i];
}
