#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#define U ((double)rand() / (double)RAND_MAX)   // draws a double number between 0 and 1
#define I(q) (((double)q)/(N+M))                // given q interval index, get interval lower bound
#define Q(x) (floor(((double)x)*(N+M)))         // given x sample, get interval index
#define R(q) ((U%I(1)) + I((q)))                // random number between I(q) and I(q+1)

typedef struct vlist{
    int len;
    int* vacancies;
} vacancies_list;

int **vacancy_space_reduction(vacancies_list*, int, int);
vacancies_list* build_vacancies_matrix(double**, int, int, int);
double** load_sample_set(PyObject*, int*, int*);
PyObject* build_result_matrix(double**, int, int);
void free_matrix(void**, int);

srand((unsigned)time(NULL)); 

//--------- CORE MODULE --------
//------------------------------
static PyObject* method_eLHS(PyObject* self, PyObject* args) {
    PyObject *matrix_object;
    // ss = sample set; exp = expansion set
    double **ss, **exp; 
    int N, M, P;

    if (!PyArg_ParseTuple(args, "Oi", &matrix_object, &M))  
        return NULL;
    
    ss = load_sample_set(matrix_object, &N, &P);
    if(!ss) return NULL;

    // METHOD'S BODY
    vacancies_list *vs = build_vacancies_matrix(ss, N, P, M);


    
    // yielding result
    // PyObject* result = build_result_matrix(ss, N, P);
    PyObject* result = build_result_matrix(ss, N, P);
    free_matrix((void**) ss, N);
    // return result;
    Py_RETURN_NONE;
}

static PyObject* method_grade(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *matrix_object; // mandatory param
    int M = 0; // optional param
    static char *kwlist[] = {"sample_set", "M", NULL};

    double **ss, gr;
    int i, j, q, N, P;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &matrix_object, &M))  
        return NULL;
    
    ss = load_sample_set(matrix_object, &N, &P);
    if(!ss) return NULL;

    
    gr = 0;
    // COMPUTE GRADE
    for(j = 0; j < P; j++){
        for(q = 0; q < N+M; q++){
            for(i = 0; i < N; i++){
                if(I(q) <= ss[i][j] && ss[i][j] < I(q+1)){
                    gr += 1;
                    break;
                }
            }
        }
    }

    gr /= (double) P * (N + M);

    free_matrix((void**) ss, N);
    return PyFloat_FromDouble(gr);
}
//-------------------------------
//ˆˆˆˆˆˆˆˆˆ CORE MODULE ˆˆˆˆˆˆˆˆˆ

int **vacancy_space_reduction(vacancies_list* vs, int M, int P){
    return NULL;
}

vacancies_list* build_vacancies_matrix(double** ss, int N, int P, int M){
    vacancies_list *vs;
    int i, j, q;

    vs = (vacancies_list*)malloc(P*sizeof(vacancies_list));

    for(j = 0; j < P; j++){
        int vindex = 0;
        int *vptr = (int*) malloc((N+M)*sizeof(int));
        bool *vmask = (bool*) malloc((N+M)*sizeof(bool));
        memset(vmask, 0, (N + M) * sizeof(bool));
        
        for(i = 0; i < N; i++)
            vmask[(int) Q(ss[i][j])] = true;
        
        for(q = 0; q < N+M; q++){
            if(!vmask[q])   // if q IS void
                vptr[vindex++] = q;
        }

        vptr = (int*) realloc(vptr, vindex * sizeof(int));
        vs[j].len = vindex;
        vs[j].vacancies = vptr;
    }
    return vs;
}

double** load_sample_set(PyObject* matrix_object, int* N, int* P){
    PyArrayObject *array = NULL;
    double **c_matrix = NULL;
    double *subarr = NULL;
    int i, j;

    /* Ensure the input is a 2D NumPy array of doubles */
    array = (PyArrayObject*) PyArray_FROM_OTF(matrix_object, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) return NULL;

    /* Check if the array is 2D */
    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected a 2D array");
        Py_DECREF(array);
        return NULL;
    }

    *N = (int) PyArray_DIM(array, 0);
    *P = (int) PyArray_DIM(array, 1);

    /* Allocate c_matrix memory*/
    c_matrix = (double**) malloc(*N * sizeof(double*));
    if (c_matrix == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for C matrix");
        Py_DECREF(array);
        return NULL;
    }

    for (i = 0; i < *N; i++) {
        c_matrix[i] = (double*) malloc(*P * sizeof(double));
        if (c_matrix[i] == NULL) {
            // if it fails, free up the memory previously allocated
            for (j = 0; j < i; j++) free(c_matrix[j]);
            free(c_matrix);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for C matrix");
            Py_DECREF(array);
            return NULL;
        }
    }

    /* Copy plane data from the numpy array to the C matrix */
    subarr = (double*) PyArray_DATA(array);
    for (i = 0; i < *N; i++) 
        for (j = 0; j < *P; j++) 
            c_matrix[i][j] = subarr[i * (*P) + j];

    /* Garbage collector syscall */
    Py_DECREF(array);

    return c_matrix;
}

PyObject* build_result_matrix(double **matrix, int M, int P){
    PyObject* result = PyList_New(M);
    int i,j;
    for (i = 0; i < M; i++) {
        PyObject* row = PyList_New(P);
        for (j = 0; j < P; j++) 
            PyList_SetItem(row, j, PyFloat_FromDouble(matrix[i][j]));
        
        PyList_SetItem(result, i, row);
    }
    return result;
}

void free_matrix(void **m, int N){
    for (int i = 0; i < N; i++) 
        free(m[i]);
    free(m);
}


// module functions to export
static PyMethodDef latinexpansionMethods[] = {
    {"eLHS", method_eLHS, METH_VARARGS, "Expansion algorithm for Latin Hypercube Sampling."},
    {"grade", (PyCFunction)method_grade, METH_VARARGS | METH_KEYWORDS, "Non-collapsing property metric for sample set."},
    {NULL, NULL, 0, NULL}
};

// module definition struct (declaring the module)
static struct PyModuleDef latinexpansionModule = {
    PyModuleDef_HEAD_INIT,
    "latinexpansion",
    "Python interface for the Latin Hypercube Sampling expansion C library and utilities",
    -1,
    latinexpansionMethods
};

// module initialization routine (ONLY declared, NOT invoked)
PyMODINIT_FUNC PyInit_latinexpansion(void) {
    import_array();  /* Initialize the NumPy C API */
    return PyModule_Create(&latinexpansionModule);
}
