#include <stdio.h>
#include "api.h"
#include "langcodes.h"
#include "rendering.h"
#include "result-codes.h"
#include "symbol-table.h"
#include "token.h"
#include "trie.h"
#include "util.h"
#include "varnam.h"
#include "varray.h"
#include "vtypes.h"
#include "vutf8.h"
#include "vword.h"
#include "words-table.h"
#include <windows.h>

extern int
stem(varnam *handle, const char *word, struct varray_t *stem_results);

extern int varnam_init(const char *scheme_file, varnam **handle, char **msg);
extern void varnam_destroy(varnam *handle);
static const char *cjf(varnam* handle, const char *s);
//const char *read_words (const char *x);

void printf_utf8(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    int len = _vscprintf(format, args) + 1; 
    char *buf = malloc(len);
    vsprintf(buf, format, args);

    //convert to UTF16 and print
    int wbuf_size = MultiByteToWideChar(CP_UTF8, 0, buf, -1, NULL, 0);
    wchar_t *wbuf = malloc((wbuf_size + 1) * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, buf, -1, wbuf, wbuf_size);

    DWORD temp;
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    WriteConsoleW(h, wbuf, wcslen(wbuf), &temp, 0);

    free(wbuf);
    free(buf);
}

int main(int argc, char **argv){
    //read_words(argv[1]);
    return 0;
}

__declspec(dllexport) void stem_it(char x[1024]) {
    varnam* handle;
    char* msg;
    int rc = varnam_init ("ml.vst", &handle, &msg);
    strcpy(x, cjf(handle,x));
    varnam_destroy (handle);
}

static const char *cjf(varnam* handle, const char *s) {
    int rc, i;
    varray *stem_results = varray_init();
    rc = stem(handle, s, stem_results);
    char *r="";
    if (stem_results->index <0) {
        return "";
    }
    return ((vword*)stem_results->memory[stem_results->index])->text;
}

