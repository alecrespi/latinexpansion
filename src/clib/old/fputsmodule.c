#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <Python.h>


//--------- CORE MODULE --------
//------------------------------
static PyObject *method_fputs(PyObject *self, PyObject *args) {
    char *str, *filename = NULL;
    int bytes_copied = -1;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "ss", &str, &filename)) {
        return NULL;
    }

    FILE *fp = fopen(filename, "w");
    bytes_copied = fputs(str, fp);
    fclose(fp);

    return PyLong_FromLong(bytes_copied);
}
//-------------------------------
//ˆˆˆˆˆˆˆˆˆ CORE MODULE ˆˆˆˆˆˆˆˆˆ


// module functions to export
static PyMethodDef FputsMethods[] = {
    {"fputs", method_fputs, METH_VARARGS, "Python interface for fputs C library function"},
    {NULL, NULL, 0, NULL}
};

// module definition struct (declaring the module)
static struct PyModuleDef fputsmodule = {
    PyModuleDef_HEAD_INIT,
    "fputs",
    "Python interface for the fputs C library function",
    -1,
    FputsMethods
};

// module initialization routine (ONLY declared, NOT invoked)
PyMODINIT_FUNC PyInit_fputs(void) {
    return PyModule_Create(&fputsmodule);
}

// // PRE-INTERPRETER CALL AND MODULE INIT
// int
// main(int argc, char *argv[])
// {
//     wchar_t *program = Py_DecodeLocale(argv[0], NULL);
//     if (program == NULL) {
//         fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
//         exit(1);
//     }

//     /* Add a built-in module, before Py_Initialize */
//     if (PyImport_AppendInittab("spam", PyInit_spam) == -1) {
//         fprintf(stderr, "Error: could not extend in-built modules table\n");
//         exit(1);
//     }

//     /* Pass argv[0] to the Python interpreter */
//     Py_SetProgramName(program);

//     /* Initialize the Python interpreter.  Required.
//        If this step fails, it will be a fatal error. */
//     Py_Initialize();

//     /* Optionally import the module; alternatively,
//        import can be deferred until the embedded script
//        imports it. */
//     PyObject *pmodule = PyImport_ImportModule("spam");
//     if (!pmodule) {
//         PyErr_Print();
//         fprintf(stderr, "Error: could not import module 'spam'\n");
//     }

//     PyMem_RawFree(program);
//     return 0;
// }