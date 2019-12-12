#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  ERROR = -2,  // abort if error
  WARNING = -1,
  INFO = 0
} VerbosityType;

#define AITISA_LOG(verbosity, message_string)                            \
  if (verbosity == ERROR) {                                              \
    fprintf(stderr, "%s: from %s(), file %s, line %d: %s\n", #verbosity,  \
            __FUNCTION__, __FILE__, __LINE__, message_string);           \
    abort();                                                             \
  } else {                                                               \
    fprintf(stdout, "%s: from %s(), file %s, line %d: %s\n", #verbosity, \
            __FUNCTION__, __FILE__, __LINE__, message_string);           \
  }

#define AITISA_CHECK(cond)                                               \
  if (!(cond)) {                                                         \
    fprintf(stderr, "ERROR: %s(), file %s, line %d: CHECK FAILED, %s\n", \
            __FUNCTION__, __FILE__, __LINE__, #cond);                    \
    abort();                                                             \
  }

#endif