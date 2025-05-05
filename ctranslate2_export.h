
#ifndef CTRANSLATE2_EXPORT_H
#define CTRANSLATE2_EXPORT_H

#ifdef CTRANSLATE2_STATIC_DEFINE
#  define CTRANSLATE2_EXPORT
#  define CTRANSLATE2_NO_EXPORT
#else
#  ifndef CTRANSLATE2_EXPORT
#    ifdef ctranslate2_EXPORTS
        /* We are building this library */
#      define CTRANSLATE2_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CTRANSLATE2_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CTRANSLATE2_NO_EXPORT
#    define CTRANSLATE2_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CTRANSLATE2_DEPRECATED
#  define CTRANSLATE2_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CTRANSLATE2_DEPRECATED_EXPORT
#  define CTRANSLATE2_DEPRECATED_EXPORT CTRANSLATE2_EXPORT CTRANSLATE2_DEPRECATED
#endif

#ifndef CTRANSLATE2_DEPRECATED_NO_EXPORT
#  define CTRANSLATE2_DEPRECATED_NO_EXPORT CTRANSLATE2_NO_EXPORT CTRANSLATE2_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef CTRANSLATE2_NO_DEPRECATED
#    define CTRANSLATE2_NO_DEPRECATED
#  endif
#endif

#endif /* CTRANSLATE2_EXPORT_H */
