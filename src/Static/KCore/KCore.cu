#include <Device/Util/Timer.cuh>
#include "Static/KCore/KCore.cuh"
#include <fstream>

#define INSERT 0
#define DELETE 1

// #include <Device/Primitives/CubWrapper.cuh>

using namespace timer;
namespace hornets_nest {

KCore::KCore(HornetGraph &hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        // src_equeue(hornet, 4.0f),
                        // dst_equeue(hornet, 4.0f),
                        peel_vqueue(hornet),
                        load_balancing(hornet) {

    h_copy_csr_off   = new vid_t[hornet.nV() + 1]();
    h_copy_csr_edges = new vid_t[0]();
    
    memset(h_copy_csr_off, 0, (hornet.nV() + 1) * sizeof(vid_t));

    gpu::allocate(vertex_pres, hornet.nV());
    gpu::allocate(vertex_subg, hornet.nV());
    gpu::allocate(hd_data().src,    hornet.nE());
    gpu::allocate(hd_data().dst,    hornet.nE());
    gpu::allocate(hd_data().src_tot,    hornet.nE());
    gpu::allocate(hd_data().dst_tot,    hornet.nE());
    gpu::allocate(hd_data().counter, 1);
    gpu::allocate(hd_data().counter_tot, 1);
    gpu::memsetZero(hd_data().counter_tot);  // initialize counter for all edge mapping.
}

KCore::~KCore() {
    gpu::free(vertex_pres);
    gpu::free(vertex_subg);
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    gpu::free(hd_data().src_tot);
    gpu::free(hd_data().dst_tot);
    gpu::free(hd_data().counter);
    gpu::free(hd_data().counter_tot);
    delete[] h_copy_csr_off;
    delete[] h_copy_csr_edges;
}

struct CheckDeg {
    TwoLevelQueue<vid_t> vqueue;
    TwoLevelQueue<vid_t> peel_vqueue;
    vid_t *vertex_pres;
    uint32_t peel;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();

        if (vertex_pres[id] && v.degree() <= peel) {
            vqueue.insert(id);
            peel_vqueue.insert(id);
            vertex_pres[id] = 0;
        }
    } 
};

struct SetPresent {
    vid_t *vertex_pres;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        vertex_pres[id] = 1;
    }
};

struct PeelVertices {
    HostDeviceVar<KCoreData> hd;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();

        int spot = atomicAdd(hd().counter, 1);
        hd().src[spot] = src;
        hd().dst[spot] = dst;
    }
};

struct Subgraph {
    HostDeviceVar<KCoreData> hd;
    vid_t *vertex_subg;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();

        if (src < dst && vertex_subg[dst] == 1) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            int spot_tot = atomicAdd(hd().counter_tot, 1);
            hd().src_tot[spot_tot] = src;
            hd().dst_tot[spot_tot] = dst;
        }
    }
};

struct SubgraphVertices {
    vid_t *vertex_subg;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vertex_subg[src] = 1;
    }
};

struct ClearSubgraph {
    vid_t *vertex_subg;

    OPERATOR(Vertex &v) {
        vid_t src = v.id();
        vertex_subg[src] = 0;
    }
};

struct PrintVertices {
    const vid_t *src_ptr;
    const vid_t *dst_ptr;
    int32_t size;

    OPERATOR(Vertex &v) {
        if (v.id() == 0) {
            for (uint32_t i = 0; i < size; i++) {
                // printf("%d ", src_ptr[i]);
                printf("batch_src[%u] = %d; batch_dst[%u] = %d;\n", i, src_ptr[i], i,
                                                                   dst_ptr[i]);
            }
        }
    }
};

void KCore::reset() {
    std::cout << "ran1" << std::endl;
}

void oper_bidirect_batch(HornetGraph &hornet, vid_t *src, vid_t *dst, 
                         int size, uint8_t op) {
    
    gpu::BatchUpdate batch_update_src(src, dst, size, gpu::BatchType::DEVICE);

    if (op == DELETE) {
        // Delete edges in the forward direction.
        hornet.deleteEdgeBatch(batch_update_src);
    } else {
        // Delete edges in the forward direction.
        hornet.insertEdgeBatch(batch_update_src);
    }

    gpu::BatchUpdate batch_update_dst(dst, src, size, gpu::BatchType::DEVICE);

    if (op == DELETE) {
        // Delete edges in reverse direction.
        hornet.deleteEdgeBatch(batch_update_dst);
    } else {
        // Delete edges in reverse direction.
        hornet.insertEdgeBatch(batch_update_dst);
    }
}

void kcores(HornetGraph &hornet, 
            HornetGraph &h_copy,
            TwoLevelQueue<vid_t> &vqueue, 
            HostDeviceVar<KCoreData>& hd, 
            TwoLevelQueue<vid_t> &peel_vqueue,
            load_balancing::VertexBased1 load_balancing,
            uint32_t *max_peel,
            vid_t *vertex_pres,
            vid_t *vertex_subg,
            uint32_t *ne,
            uint32_t iter_count) {

    uint32_t peel = 0;
    uint32_t nv = hornet.nV();
    int size = 0;

    while (nv > 0) {
        forAllVertices(hornet, CheckDeg { vqueue, peel_vqueue, 
                                          vertex_pres, peel });
        
        vqueue.swap();
        nv -= vqueue.size();
        
        // vqueue.print();

        if (vqueue.size() > 0) {
            // Find all vertices with degree <= peel.
            gpu::memsetZero(hd().counter);  // reset counter. 

            forAllEdges(hornet, vqueue, PeelVertices { hd }, load_balancing); 

            cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);

            if (size > 0) {
                oper_bidirect_batch(hornet, hd().src, hd().dst, size, DELETE);
                oper_bidirect_batch(h_copy, hd().src, hd().dst, size, INSERT);
            }

            *ne -= 2 * size;

            vqueue.clear();
        } else {
            peel++;    
            peel_vqueue.swap();
        }
    }
    *max_peel = peel;

    peel_vqueue.swap();

    forAllEdges(h_copy, peel_vqueue, SubgraphVertices { vertex_subg }, load_balancing);

    gpu::memsetZero(hd().counter);  // reset counter. 
    forAllEdges(h_copy, peel_vqueue, Subgraph { hd, vertex_subg }, load_balancing);

    forAllVertices(h_copy, ClearSubgraph { vertex_subg });
    
    cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (size > 0) {
        oper_bidirect_batch(h_copy, hd().src, hd().dst, size, DELETE);
    }
}

HornetGraph* hornet_copy(HornetGraph &hornet,
                         vid_t *h_copy_csr_off,
                         vid_t *h_copy_csr_edges) {
                         // TwoLevelQueue<vid_t> tot_src_equeue,
                         // TwoLevelQueue<vid_t> tot_dst_equeue) {

    HornetInit hornet_init(hornet.nV(), 0, h_copy_csr_off,
                           h_copy_csr_edges, false);

    HornetGraph *h_copy = new HornetGraph(hornet_init);

    return h_copy;
}

void json_dump(vid_t *src, vid_t *dst, uint32_t *peel, uint32_t peel_edges) {
    std::ofstream output_file;
    output_file.open("output.txt");
    
    output_file << "{\n";
    for (uint32_t i = 0; i < peel_edges; i++) {
        output_file << "\"" << src[i] << "," << dst[i] << "\": " << peel[i];
        if (i < peel_edges - 1) {
            output_file << ",";
        }
        output_file << "\n";
    }
    output_file << "}";
    output_file.close();
}

void KCore::run() {
    omp_set_num_threads(72);
    vid_t *src     = new vid_t[hornet.nE() / 2 + 1];
    vid_t *dst     = new vid_t[hornet.nE() / 2 + 1];
    uint32_t len = hornet.nE() / 2 + 1;
    uint32_t *peel = new uint32_t[hornet.nE() / 2 + 1];
    uint32_t peel_edges = 0;
    uint32_t ne = hornet.nE();
    uint32_t ne_orig = hornet.nE();

    auto pres = vertex_pres;
    auto subg = vertex_subg;
    
    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 1; } );
    forAllnumV(hornet, [=] __device__ (int i){ subg[i] = 0; } );

    Timer<DEVICE> TM;
    TM.start();
    HornetGraph &h_copy = *hornet_copy(hornet, h_copy_csr_off,
                                       h_copy_csr_edges);

    uint32_t iter_count = 0; 
    int size = 0;

    while (peel_edges < ne_orig / 2) {
        uint32_t max_peel = 0;
        ne = ne_orig - 2 * peel_edges;

        if (iter_count % 2) {
            kcores(h_copy, hornet, vqueue, hd_data, peel_vqueue, 
                   load_balancing, &max_peel, vertex_pres, vertex_subg, &ne, iter_count);
            
            forAllVertices(hornet, SetPresent { vertex_pres });
        } else {
            kcores(hornet, h_copy, vqueue, hd_data, peel_vqueue, 
                   load_balancing, &max_peel, vertex_pres, vertex_subg, &ne, iter_count);

            forAllVertices(h_copy, SetPresent { vertex_pres });
        }

        
        std::cout << "max_peel: " << max_peel << "\n";

        cudaMemcpy(&size, hd_data().counter, sizeof(int), 
                   cudaMemcpyDeviceToHost);

        if (size > 0) {
            #if 0
            cudaMemcpy(src + peel_edges, hd_data().src, 
                       size * sizeof(vid_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(dst + peel_edges, hd_data().dst, 
                       size * sizeof(vid_t), cudaMemcpyDeviceToHost);
            #endif

            #pragma omp parallel for
            for (uint32_t i = 0; i < size; i++) {
                peel[peel_edges + i] = max_peel;
            }

            peel_edges += size;
        }

        iter_count++;

        if (peel_edges >= len) {
            std::cout << "ooooops" << std::endl;
            std::cout << "peel_edges " << peel_edges << " len " << len << std::endl;
        }
    }
    TM.stop();
    TM.print("KCore");

    cudaMemcpy(src, hd_data().src_tot, 
               peel_edges * sizeof(vid_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(dst, hd_data().dst_tot, 
                peel_edges * sizeof(vid_t), cudaMemcpyDeviceToHost);

    json_dump(src, dst, peel, peel_edges);
}


void KCore::release() {
    std::cout << "ran3" << std::endl;
}
}
