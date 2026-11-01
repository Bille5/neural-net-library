#ifdef _WIN32

#include <stdint.h>
#include <stdlib.h>
#include <windows.h>
#include <bcrypt.h>

uint32_t arc4random(void) {
    uint32_t value;
    NTSTATUS status = BCryptGenRandom(
        NULL,
        (PUCHAR)&value,
        sizeof(value),
        BCRYPT_USE_SYSTEM_PREFERRED_RNG
    );

    if (status < 0) {
        abort();
    }

    return value;
}

#endif
