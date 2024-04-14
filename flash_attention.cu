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
    int q_offset = (bz*gridDim.y*gridDim.x*N*d+by*girdDim.x*N*d);
    int kv_offset = (bz*gridDim.y*gridDim.x*N*d + by*gridDim.x*N*d );  
    int lm_offset = (bz*gridDim.y*gridDim.x*N+by*gridDim.x*N);
;
    //定义并分配sharing memory
    //其中Q要求分块加载但是必须要并行，所以要分配足够大的空间
    //由于K,V可以分块，所以给tile_size的空间足够
    //由于是并行所以S的空间也必须给够
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    int tile_s_size =N * Br;  
    float* Qi = sram;
    float* Kj = &sram[tile_size* Tc];
    float* Vj = &sram[tile_size *(Tc + 1) ];
    float* S = &sram[tile_size * (Tc + 1 ) + tile_s_size];
    float* O = &sram[tile_size * (Tc + ) + tile_s_size]
        //采用Tr并行的方式加载Q
        for(int x=0;x<d;x++){
            //Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            Qi[blockIdx.x * tile_size + (tx * d) + x] =Q[q_offset +tile_size * blockIdx.x + (tx * d) + x];
        }

        //load Kj Vj TO SRAM
        float  row_m_prev = m[lm_offset+blockIdx.x*Bc+tx];
        float  row_l_prev = l[lm_offset+blockIdx.x*Bc+tx];
        for(int j=0;j<Tr;j++){
            //加载K,V到SRAM中
            float sum =0;
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[kv_offset + (tile_size * j) + (tx * d) + x];
            //Vj[(tx * d) + x] = V[kv_offset + (tile_size * j) + (tx * d) + x];
            sum += Qi[blockIdx.x * tile_size + (tx * d) + x] * Kj[(tx * d) + x];
        }
       	    S[blockIdx.x * Br * Bc + (tx * d)] = sum ;
            //计算S的值
            //计算O的值
        for (int x = 0; x < d; x++) {
            //Kj[(tx * d) + x] = K[kv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[kv_offset + (tile_size * j) + (tx * d) + x];
            O[blockIdx.x*tile_size+(tx * d) + x]=S[blockIdx.x * Br * Bc + ( tx * Br) + x] * Vj[(tx * d) + x];
        }       
        } 

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


    const int sram_size = ( 2 * N * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (N * Br * sizeof(float));
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
