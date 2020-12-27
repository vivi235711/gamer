#include "CUFLU.h"



// external functions and GPU-related set-up
#ifdef __CUDACC__

#if ( MODEL == HYDRO )
#include "CUFLU_Shared_FluUtility.cu"
#endif
#include "CUDA_ConstMemory.h"
#include "Deleptonization/GPU_Src_Deleptonization.cu"

#else

void Src_Deleptonization( real fluid[], const real B[],
                          const SrcTerms_t SrcTerms, const real dt, const real dh,
                          const double x, const double y, const double z,
                          const double TimeNew, const double TimeOld,
                          const real MinDens, const real MinPres, const real MinEint,
                          const EoS_DE2P_t EoS_DensEint2Pres,
                          const EoS_DP2E_t EoS_DensPres2Eint,
                          const EoS_DP2C_t EoS_DensPres2CSqr,
                          const double EoS_AuxArray_Flt[],
                          const int    EoS_AuxArray_Int[],
                          const real *const EoS_Table[EOS_NTABLE_MAX] );

#endif // #ifdef __CUDACC__ ... else ...




//-------------------------------------------------------------------------------------------------------
// Function    :  CPU/GPU_SrcSolver_IterateAllCells
// Description :  Iterate over all cells to add each source term
//
// Note        :  1. Invoked by CPU_SrcSolver() and CUAPI_Asyn_SrcSolver()
//                2. No ghost zones
//                   --> Should support ghost zones in the future
//
// Parameter   :  g_Flu_Array_In         : Array storing the input fluid variables
//                g_Flu_Array_Out        : Array to store the output fluid variables
//                g_Mag_Array_In         : Array storing the input B field (for MHD only)
//                g_Corner_Array         : Array storing the physical corner coordinates of each patch
//                SrcTerms               : Structure storing all source-term variables
//                NPatchGroup            : Number of patch groups to be evaluated
//                dt                     : Time interval to advance solution
//                dh                     : Grid size
//                TimeNew                : Target physical time to reach
//                TimeOld                : Physical time before update
//                                         --> This function updates physical time from TimeOld to TimeNew
//                MinDens/Pres/Eint      : Density, pressure, and internal energy floors
//                EoS_DensEint2Pres_Func : Function pointer to the EoS routine of computing the gas pressure
//                EoS_DensPres2Eint_Func :                    . . .                             gas internal energy
//                EoS_DensPres2CSqr_Func :                    . . .                             sound speed square
//                c_EoS_AuxArray_*       : Auxiliary arrays for the EoS routines (for CPU only)
//                c_EoS_Table            : EoS tables                            (for CPU only)
//                                         --> When using GPU, these CPU-only variables are stored in the constant memory
//                                             header CUDA_ConstMemory.h and do not need to be passed as function arguments
//
// Return      : fluid[] in all patches
//-------------------------------------------------------------------------------------------------------
#ifdef __CUDACC__
__global__
void CUSRC_SrcSolver_IterateAllCells(
   const real g_Flu_Array_In [][FLU_NIN_S ][ CUBE(SRC_NXT)           ],
         real g_Flu_Array_Out[][FLU_NOUT_S][ CUBE(SRC_NXT)           ],
   const real g_Mag_Array_In [][NCOMP_MAG ][ SRC_NXT_P1*SQR(SRC_NXT) ],
   const double g_Corner_Array[][3],
   const SrcTerms_t SrcTerms, const int NPatchGroup, const real dt, const real dh,
   const double TimeNew, const double TimeOld,
   const real MinDens, const real MinPres, const real MinEint,
   const EoS_DE2P_t EoS_DensEint2Pres_Func,
   const EoS_DP2E_t EoS_DensPres2Eint_Func,
   const EoS_DP2C_t EoS_DensPres2CSqr_Func )
#else
void CPU_SrcSolver_IterateAllCells(
   const real g_Flu_Array_In [][FLU_NIN_S ][ CUBE(SRC_NXT)           ],
         real g_Flu_Array_Out[][FLU_NOUT_S][ CUBE(SRC_NXT)           ],
   const real g_Mag_Array_In [][NCOMP_MAG ][ SRC_NXT_P1*SQR(SRC_NXT) ],
   const double g_Corner_Array[][3],
   const SrcTerms_t SrcTerms, const int NPatchGroup, const real dt, const real dh,
   const double TimeNew, const double TimeOld,
   const real MinDens, const real MinPres, const real MinEint,
   const EoS_DE2P_t EoS_DensEint2Pres_Func,
   const EoS_DP2E_t EoS_DensPres2Eint_Func,
   const EoS_DP2C_t EoS_DensPres2CSqr_Func,
   const double c_EoS_AuxArray_Flt[],
   const int    c_EoS_AuxArray_Int[],
   const real* const c_EoS_Table[EOS_NTABLE_MAX] )
#endif
{

// loop over all patches
// --> CPU/GPU solver: use different (OpenMP threads) / (CUDA thread blocks)
//     to work on different patches
#  ifdef __CUDACC__
   const int p = blockIdx.x;
#  else
#  pragma omp parallel for schedule( runtime )
   for (int p=0; p<8*NPatchGroup; p++)
#  endif
   {
      const double x0 = g_Corner_Array[p][0] - SRC_GHOST_SIZE*dh;
      const double y0 = g_Corner_Array[p][1] - SRC_GHOST_SIZE*dh;
      const double z0 = g_Corner_Array[p][2] - SRC_GHOST_SIZE*dh;

//###REVISE: support ghost zones
      CGPU_LOOP( t, CUBE(SRC_NXT) )
      {
//       compute the cell-centered coordinates
         double x, y, z;
         int    i, j, k;

         i = t % SRC_NXT;
         j = t % SQR(SRC_NXT) / SRC_NXT;
         k = t / SQR(SRC_NXT);

         x = x0 + double(i*dh);
         y = y0 + double(j*dh);
         z = z0 + double(k*dh);


//       get the input arrays
         real fluid[FLU_NIN_S];

         for (int v=0; v<FLU_NIN_S; v++)  fluid[v] = g_Flu_Array_In[p][v][t];

#        ifdef MHD
         real B[NCOMP_MAG];
         MHD_GetCellCenteredBField( B, g_Mag_Array_In[p][MAGX], g_Mag_Array_In[p][MAGY], g_Mag_Array_In[p][MAGZ],
                                    SRC_NXT, SRC_NXT, SRC_NXT, i, j, k );
#        else
         real *B = NULL;
#        endif


//       add all source terms one by one
//       (1) deleptonization
         if ( SrcTerms.Deleptonization )
            Src_Deleptonization  ( fluid, B, SrcTerms, dt, dh, x, y, z, TimeNew, TimeOld, MinDens, MinPres, MinEint,
                                   EoS_DensEint2Pres_Func, EoS_DensPres2Eint_Func, EoS_DensPres2CSqr_Func,
                                   c_EoS_AuxArray_Flt, c_EoS_AuxArray_Int, c_EoS_Table );

//       (2) user-defined
         if ( SrcTerms.User )
            SrcTerms.User_FuncPtr( fluid, B, SrcTerms, dt, dh, x, y, z, TimeNew, TimeOld, MinDens, MinPres, MinEint,
                                   EoS_DensEint2Pres_Func, EoS_DensPres2Eint_Func, EoS_DensPres2CSqr_Func,
                                   c_EoS_AuxArray_Flt, c_EoS_AuxArray_Int, c_EoS_Table );

//       store the updated results
         for (int v=0; v<FLU_NOUT_S; v++)   g_Flu_Array_Out[p][v][t] = fluid[v];

      } // CGPU_LOOP( t, CUBE(SRC_NXT) )
   } // for (int p=0; p<8*NPatchGroup; p++)

} // FUNCTION : CPU/GPU_SrcSolver_IterateAllCells
