#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O){
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (by*gridDim.z*N*d + bz*N*d);
    //int kv_offset = (bz*gridDim.y*gridDim.x*N*d + by*gridDim.x*N*d );
    int kv_offset = (by*gridDim.z*N*d + bz*N*d);  
    int lm_offset = (by*gridDim.y*N+by*N);

    extern __shared__ float sram[];
    int tile_q_size = Bc * d;
    int tile_s_size = Bc * Br; 
    int tile_kv_size =Br * d; 
    float* Qi = sram;
    float* Kj = &sram[tile_q_size];
    float* Vj = &sram[tile_q_size + tile_kv_size ];
    float* Si = &sram[tile_q_size + (tile_kv_size *2)];
    float* Oi = &sram[tile_q_size + (tile_kv_size *2) + tile_s_size];
        //采用Tr并行的方式加载Q
        for(int x=0;x<d;x++){
            //Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            Qi[(tx * d) + x] =Q[q_offset +tile_q_size * blockIdx.x + (tx * d) + x];
        }

        float  row_m_prev = m[lm_offset+blockIdx.x*Bc+tx];
        float  row_l_prev = l[lm_offset+blockIdx.x*Bc+tx];
        float  row_l_max  =  0;
        for(int j=0;j<Tr;j++){
        float  row_m = -INFINITY;
        //加载K,V到SRAM中,compute Si
        for(int mi = 0;mi<Br;mi++ ){
        float sum =0;
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[kv_offset + (tile_kv_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[kv_offset + (tile_kv_size * j) + (tx * d) + x];

            if(j>0){
            Oi[(tx * d) + x] = O[q_offset + (tile_q_size *(j-1))+ (tx * d) + x];
            }else{
            Oi[(tx * d) + x] = O[q_offset + (tile_q_size *j)+ (tx * d) + x];
            }

            //Vj[(tx * d) + x] = V[kv_offset + (tile_size * j) + (tx * d) + x];
            sum += Qi[(tx * d) + x] * Kj[(tx * d) + x];
        }
       	    Si[(tx * d) + mi] = sum ;

            // find row max
            if(sum > row_m){
                row_m =sum;
            }
        }

        float row_m_new = max(row_m_prev,row_m);
        int sum_new = 0;

        for(int x=0;x<Br;x++){
            Si[(tx * Br) + x] = __expf(Si[(tx * Br) + x] - row_m_new);
            sum_new += Si[(tx * Br) + x] ;
        }
        //l2_new
        float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + sum_new ; 

        for(int n =0;n < d;n++){
            float pv=0; 
            for(int x = 0; x<Br; x++){
                pv += Si[(tx * Br) + x] * Vj[(x * d) + n]; 
        }
           Oi[(tx * d)+ n ] =__expf(row_m_prev - row_m_new) * Oi[(tx *d) + n ]+ pv ;
        }
        m[lm_offset + (Bc * blockIdx.x) + tx] = row_m_new;
        l[lm_offset + (Bc * blockIdx.x) + tx ] = row_l_new;
        
        row_l_max = row_l_new;
        } 

        for(int i =0 ;i <d;i++){
            Oi[(tx * d) + i] = Oi[(tx * d) + i] * (1/row_l_max);
            O[q_offset + (tile_q_size * blockIdx.x)+(tx *d) + i] = Oi[(tx + d) + i];
        }
        


        __syncthreads();
    }




torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32; 
    const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);


    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);
	
    
    //const int sram_size = ( 2 * N * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (N * Br * sizeof(float));

    const int sram_size = (2 * Bc * d * sizeof(float)) + (2 * Br * d * sizeof(float)) + (Bc * Br * sizeof(float));

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(Tc , B, nh); 
    dim3 block_dim(Bc);  

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}

