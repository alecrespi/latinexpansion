#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#define I(q) (((double)q)/(N+M))                // given q interval index, get interval lower bound
#define Q(x) (floor(((double)x)*(N+M)))         // given x sample, get interval index
#define SWAP(a, b) do { typeof(a) tmp = a; a = b; b = tmp; } while (0)  // var swapping
#define R(q) ((double)I(q) + ((double)rand() * I(1) / RAND_MAX))

typedef struct vlist{
    int len;
    int* vacancies;
} vacancies_list;

double euclid(double*, double*, int);
double min_distance(double**, double**, int, int, int);
void print_double_matrix(double**, int, int);
void print_int_matrix(int**, int, int);
int **shuffle_subset_vacancies(vacancies_list*, int, int);
vacancies_list* build_vacancies_matrix(double**, int, int, int);
double** load_sample_set(PyObject*, int*, int*);
PyObject* build_result_matrix(double**, int, int);
void free_matrix(void**, int);

//--------- CORE MODULE --------
//------------------------------
static PyObject* method_eLHS(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *matrix_object;
    vacancies_list *vs;
    int **v_prime; 
    double **ss, **exp; /* sample set, expansion set */
    double max_distance, dist;
    int i, j, t, N, M, P, throws = 10, criteria = 1;
    // criteria = 0 : RANDOM ; criteria = 1 : maximize minimum distance
    static char *kwlist[] = {"sample_set", "M", "throws", "criteria", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", kwlist, &matrix_object, &M, &throws, &criteria))
        return NULL;

    if(M == 0)
        return build_result_matrix(exp, 0, 0);
    
    ss = load_sample_set(matrix_object, &N, &P);
    if(!ss) return NULL;

    if(criteria == 0)
        throws = 1;

    // METHOD'S BODY
    exp = (double **)malloc(M * sizeof(double*));
    for(i = 0; i < M; i++)
        exp[i] = (double *)malloc(P * sizeof(double));

    vs = build_vacancies_matrix(ss, N, P, M);

    // maximin distance
    max_distance = 0.0;
    for(t = 0; t < throws; t++){
        // shuffle and subset vacancies set
        v_prime = shuffle_subset_vacancies(vs, M, P);
        
        // sowing new points
        for(i = 0; i < M; i++)
            for(j = 0; j < P; j++)
                exp[i][j] = R(v_prime[j][i]);
        
        if(throws != 1){
            dist = min_distance(ss, exp, N, M, P);
            if(dist > max_distance)
                max_distance = dist;
        }
    }

    // handling result
    PyObject* result = build_result_matrix(exp, M, P);
    free(vs);
    free_matrix((void**) v_prime, P);
    free_matrix((void**) ss, N);
    return result;
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

double euclid(double* p1, double* p2, int P) {
    double sum = 0.0;
    for (int i = 0; i < P; i++) 
        sum += pow(p1[i] - p2[i], 2);
    return sqrt(sum);
}

double min_distance(double **ss, double **exp, int N, int M, int P){
    int i, k;
    double min = __DBL_MAX__, dist = 0.0;
    for(k = 0; k < M; k++){
        for(i = 0; i < N; i++){
            dist = euclid(exp[k], ss[i], P);
            if( dist < min ) min = dist;
        }
        for(i = k + 1; i < M; i++){
            dist = euclid(exp[k], exp[i], P);
            if( dist < min ) min = dist;
        }
    }
    return min;
}

int** transpose_voids(int** original, int P, int M) {
    int **transposed, i, j;
    transposed = (int **)malloc(M * sizeof(int*));
    for (i = 0; i < M; i++) 
        transposed[i] = (int *)malloc(P * sizeof(int));

    for (j = 0; j < P; j++) 
        for (i = 0; i < M; i++) 
            transposed[i][j] = original[j][i];
        
    return transposed;
}

void print_double_matrix(double **m, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++)
            printf("%.12lf |", m[i][j]);
        printf("\n");
    }
}
void print_int_matrix(int **m, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++)
            printf("%d |", m[i][j]);
        printf("\n");
    }
}

// subsets vs rows to M elements and shuffle them
int **shuffle_subset_vacancies(vacancies_list* vs, int M, int P){
    if(!vs || M == 0) return NULL;
    int **v_prime = (int **)malloc(P * sizeof(int*));

    /* for each dimension, shuffle and reshape the vacancy list to a regular size M */
    for(int j = 0; j < P; j++){
        if(!vs[j].vacancies || !vs[j].len)
            return NULL;

        int *vptr = (int*)malloc(vs[j].len * sizeof(int));
        /* copy elements to preserve data integrity for further steps*/
        for(int i = 0; i < vs[j].len; i++)
            vptr[i] = vs[j].vacancies[i];

        /* shuffling all items */
        for (int i = vs[j].len; i > 0; i--){ 
            // generate a random index that ranges from [0, M-1]
            int r = rand() % i;     
            SWAP(vptr[i-1], vptr[r]);
        }
        /* truncate the list to M */
        vptr = (int *)realloc(vptr, M*sizeof(int)); 
        v_prime[j] = vptr;
    }
    return v_prime;
}

vacancies_list* build_vacancies_matrix(double** ss, int N, int P, int M){
    vacancies_list *vs;
    int i, j, q;

    // setting up space for result
    vs = (vacancies_list*)malloc(P*sizeof(vacancies_list));

    for(j = 0; j < P; j++){
        int vindex = 0;
        int *vptr = (int*) malloc((N+M)*sizeof(int));
        /* masks the void (if index is FALSE) indexes of vptr  */
        bool *vmask = (bool*) malloc((N+M)*sizeof(bool));   
        /* set mask to false */
        memset(vmask, 0, (N + M) * sizeof(bool));           
        
        // map down the busy intervals (busy is TRUE)
        for(i = 0; i < N; i++)
            vmask[(int) Q(ss[i][j])] = true;
        
        // mask the interval indexes to gather only voids
        for(q = 0; q < N+M; q++){
            // if q is a void, then add it
            if(!vmask[q])   
                vptr[vindex++] = q;
        }

        // realloc voids sequence to fit void numbers
        vptr = (int*) realloc(vptr, vindex * sizeof(int));
        if(!vptr)   return NULL;
        // build up vacancies_list entry
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

PyObject* build_result_matrix(double **matrix, int nrows, int ncols){
    PyObject* result = PyList_New(nrows);
    int i,j;

    for (i = 0; i < nrows; i++) {
        PyObject* row = PyList_New(ncols);
        for (j = 0; j < ncols; j++) 
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
    {"eLHS", (PyCFunction)method_eLHS, METH_VARARGS | METH_KEYWORDS, "Expansion algorithm for Latin Hypercube Sampling."},
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
    srand((unsigned int)time(NULL));
    import_array();  /* Initialize the NumPy C API */
    return PyModule_Create(&latinexpansionModule);
}
