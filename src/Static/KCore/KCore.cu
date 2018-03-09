#include "Static/KCore/KCore.cuh"
#include <fstream>

#define INSERT 0
#define DELETE 1
// #include <Device/Primitives/CubWrapper.cuh>

namespace hornets_nest {

KCore::KCore(HornetGraph &hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        src_equeue(hornet),
                        dst_equeue(hornet),
                        peel_vqueue(hornet),
                        load_balancing(hornet) {

    h_copy_csr_off   = new vid_t[hornet.nV() + 1]();
    h_copy_csr_edges = new vid_t[0]();
    
    memset(h_copy_csr_off, 0, (hornet.nV() + 1) * sizeof(vid_t));

    gpu::allocate(vertex_pres, hornet.nV());
    // memset(h_copy_csr_edges, 0, hornet.nE() * sizeof(vid_t));
}

KCore::~KCore() {
    gpu::free(vertex_pres);
    // gpu::free(h_copy_csr_off);
    // gpu::free(h_copy_csr_edges);
    // delete[] h_copy_csr_off;
    // delete[] h_copy_csr_edges;
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
    // HostDeviceVar<KCoreData> hd;
    TwoLevelQueue<vid_t> src_equeue;
    TwoLevelQueue<vid_t> dst_equeue;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();
        #if 0
        int counter = hd().counter;
        hd().src[counter] = src;
        hd().dst[counter] = dst;
        atomicAdd(&(hd().counter), 1);
        #endif
        src_equeue.insert(src);
        dst_equeue.insert(dst);
    }
};

struct RemoveDuplicates {
    TwoLevelQueue<vid_t> src_equeue;
    TwoLevelQueue<vid_t> dst_equeue;
    const vid_t *src_ptr;
    const vid_t *dst_ptr;
    int32_t size;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();
        
        uint8_t double_exists = 0;
        if (src < dst) {
            for (uint32_t i = 0; i < size; i++) {
                if (src_ptr[i] == dst && dst_ptr[i] == src) {
                    double_exists = 1;
                    break;
                }
            }
        }

        if (!double_exists) {
            src_equeue.insert(src);
            dst_equeue.insert(dst);
        }
    }

};

struct Subgraph {
    TwoLevelQueue<vid_t> src_equeue;
    TwoLevelQueue<vid_t> dst_equeue;
    const vid_t *peelq_ptr;
    int32_t size;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        auto dst = e.dst_id();

        uint8_t exists = 0;
        if (src < dst) {
            for (uint32_t i = 0; i < size; i++) {
                if (peelq_ptr[i] == dst) {
                    exists = 1;
                    break;
                }
            }
        }

        if (exists){
            src_equeue.insert(src);
            dst_equeue.insert(dst);
        }
    }
};

struct PrintVertices {
    const vid_t *src_ptr;
    const vid_t *dst_ptr;
    int32_t size;

    OPERATOR(Vertex &v) {
        if (v.id() == 0) {
            for (uint32_t i = 0; i < size; i++) {
                // printf("%d %d\n", src_ptr[i], dst_ptr[i]);
                printf("batch_src[%d] = %d; batch_dst[%d] = %d;\n", i, src_ptr[i], i, dst_ptr[i]);
            }
        }
    }
};

void KCore::reset() {
    std::cout << "ran1" << std::endl;
}

void oper_bidirect_batch(HornetGraph &hornet,
                           TwoLevelQueue<vid_t> src_equeue,
                           TwoLevelQueue<vid_t> dst_equeue,
                           uint8_t op) {


    #if 0
    std::cout << "oper_og: " << unsigned(op) << "\n";
    hornet.print();
    std::cout << "\n\n";
    #endif

    // Sort src_equeue, dst_equeue by src vertex.
    xlib::CubSortPairs2<vid_t, vid_t>::srun(
                       (vid_t*) src_equeue.device_input_ptr(),
                       (vid_t*) dst_equeue.device_input_ptr(),
                                src_equeue.size(),
                        (vid_t) std::numeric_limits<vid_t>::max(),
                        (vid_t) std::numeric_limits<vid_t>::max());

    gpu::BatchUpdate batch_update_src(
                       (vid_t*) src_equeue.device_input_ptr(),
                       (vid_t*) dst_equeue.device_input_ptr(),
                                src_equeue.size());

    #if 0
    if (op == DELETE) {
        std::cout << "sorted by src " << src_equeue.size() << std::endl;
        forAllVertices(hornet, PrintVertices { src_equeue.device_input_ptr(),
                                               dst_equeue.device_input_ptr(),
                                               src_equeue.size() } );
        std::cout << "\n\n";
    }
    #endif

    if (op == DELETE) {
        #if 0
        hornet.allocateEdgeDeletion(src_equeue.size(), 
                                    gpu::batch_property::IN_PLACE);
        #endif

        // Delete edges in the forward direction.
        hornet.deleteEdgeBatch(batch_update_src);
    } else {
        #if 0
        hornet.allocateEdgeInsertion(src_equeue.size(), 
                                gpu::batch_property::IN_PLACE);
                                // gpu::batch_property::REMOVE_CROSS_DUPLICATE);
        #endif

        // Delete edges in the forward direction.
        hornet.insertEdgeBatch(batch_update_src);
    }

    #if 0
    std::cout << "oper_src: " << unsigned(op) << "\n";
    hornet.print();
    std::cout << "\n\n";
    #endif

    // Sort src_equeue, dst_equeue by dst vertex.
    xlib::CubSortPairs2<vid_t, vid_t>::srun(
                       (vid_t*) dst_equeue.device_input_ptr(),
                       (vid_t*) src_equeue.device_input_ptr(),
                                src_equeue.size(),
                        (vid_t) std::numeric_limits<vid_t>::max(),
                        (vid_t) std::numeric_limits<vid_t>::max());

    gpu::BatchUpdate batch_update_dst(
                       (vid_t*) dst_equeue.device_input_ptr(),
                       (vid_t*) src_equeue.device_input_ptr(),
                                src_equeue.size());

    #if 0
    if (op == DELETE) {
        std::cout << "sorted by dst " << src_equeue.size() << std::endl;
        forAllVertices(hornet, PrintVertices { src_equeue.device_input_ptr(),
                                               dst_equeue.device_input_ptr(),
                                               src_equeue.size() } );
        std::cout << "\n\n";
    }
    #endif

    if (op == DELETE) {
        #if 0
        hornet.allocateEdgeDeletion(src_equeue.size(), 
                                    gpu::batch_property::IN_PLACE);
        #endif

        // Delete edges in reverse direction.
        hornet.deleteEdgeBatch(batch_update_dst);
    } else {
        #if 0
        hornet.allocateEdgeInsertion(src_equeue.size(), 
                                gpu::batch_property::IN_PLACE); 
                                // gpu::batch_property::REMOVE_CROSS_DUPLICATE);
        #endif

        // Delete edges in reverse direction.
        hornet.insertEdgeBatch(batch_update_dst);
    }

    #if 0
    std::cout << "oper_dst: " << unsigned(op) << "\n";
    hornet.print();
    std::cout << "\n\n";
    #endif
}

void kcores(HornetGraph &hornet, 
            HornetGraph &h_copy,
            TwoLevelQueue<vid_t> &vqueue, 
            TwoLevelQueue<vid_t> &src_equeue,
            TwoLevelQueue<vid_t> &dst_equeue,
            TwoLevelQueue<vid_t> &peel_vqueue,
            load_balancing::VertexBased1 load_balancing,
            uint32_t *max_peel,
            vid_t *vertex_pres,
            uint32_t *ne) {

    uint32_t peel = 0;
    uint32_t nv = hornet.nV();
    // hornet.print();

    // while (*ne > 0) {
    while (nv > 0) {
        forAllVertices(hornet, CheckDeg { vqueue, peel_vqueue, 
                                          vertex_pres, peel });
        
        vqueue.swap();
        nv -= vqueue.size();
        
        // vqueue.print();

        if (vqueue.size() > 0) {
            // Find all vertices with degree <= peel.
            forAllEdges(hornet, vqueue, 
                        PeelVertices { src_equeue, dst_equeue }, 
                        load_balancing); 

            src_equeue.swap();
            dst_equeue.swap();

            // Remove duplicate edges in src_equeue and dst_equeue
            // (can happen if two vertices in vqueue are neighbors).
            forAllEdges(hornet, vqueue,
                        RemoveDuplicates { src_equeue,
                                           dst_equeue,
                                           src_equeue.device_input_ptr(),
                                           dst_equeue.device_input_ptr(),
                                           src_equeue.size() },
                        load_balancing);

            src_equeue.swap();
            dst_equeue.swap();


            if (src_equeue.size() > 0) {
                // src_equeue.print();
                // dst_equeue.print();

                oper_bidirect_batch(hornet, src_equeue, dst_equeue, DELETE);
                // hornet.print();
                oper_bidirect_batch(h_copy, src_equeue, dst_equeue, INSERT);
            }

            *ne -= 2 * src_equeue.size();

            // Save vqueue if ne == 0 -- these are vertices in the kcore.
            //if (*ne > 0) {
            vqueue.clear();
            //}
        } else {
            peel++;    
            peel_vqueue.swap();
        }
    }
    *max_peel = peel;

    peel_vqueue.swap();
    peel_vqueue.print();

    forAllEdges(h_copy, peel_vqueue,
                Subgraph { src_equeue,
                           dst_equeue,
                           peel_vqueue.device_input_ptr(),
                           peel_vqueue.size() },
                load_balancing);
    
    src_equeue.swap();
    dst_equeue.swap();

    if (src_equeue.size() > 0) {
        oper_bidirect_batch(h_copy, src_equeue, dst_equeue, DELETE);
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
    vid_t *src     = new vid_t[hornet.nE() / 2 + 1];
    vid_t *dst     = new vid_t[hornet.nE() / 2 + 1];
    uint32_t *peel = new uint32_t[hornet.nE() / 2 + 1];
    uint32_t peel_edges = 0;
    uint32_t ne = hornet.nE();
    uint32_t ne_orig = hornet.nE();

    auto pres = vertex_pres;
    
    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 1; } );

    HornetGraph &h_copy = *hornet_copy(hornet, h_copy_csr_off,
                                       h_copy_csr_edges);

    #if 0
    std::cout << "hornet:\n";
    hornet.print();
    std::cout << "\n\n";

    std::cout << "h_copy:\n";
    h_copy.print();
    std::cout << "\n\n";
    #endif

    uint32_t iter_count = 0; 
    while (peel_edges < ne_orig / 2) {
        uint32_t max_peel = 0;
        ne = ne_orig - 2 * peel_edges;

        #if 0
        std::cout << "hornet:\n";
        hornet.print();
        std::cout << "\n\n";

        std::cout << "h_copy:\n";
        h_copy.print();
        std::cout << "\n\n";
        #endif

        if (iter_count % 2) {
            kcores(h_copy, hornet, vqueue, src_equeue, dst_equeue, 
                   peel_vqueue, load_balancing, &max_peel, vertex_pres, &ne);
            
            forAllVertices(hornet, SetPresent { vertex_pres });
        } else {
            kcores(hornet, h_copy, vqueue, src_equeue, dst_equeue, 
                   peel_vqueue, load_balancing, &max_peel, vertex_pres, &ne);

            forAllVertices(h_copy, SetPresent { vertex_pres });
        }

        #if 0
        std::cout << "hornet:\n";
        hornet.print();
        std::cout << "\n\n";

        std::cout << "h_copy:\n";
        h_copy.print();
        std::cout << "\n\n";
        #endif


        // vqueue.print();
        
        std::cout << "max_peel: " << max_peel << "\n";
        src_equeue.print();
        dst_equeue.print();

        if (src_equeue.size() > 0) {
            cudaMemcpy(src + peel_edges, src_equeue.device_input_ptr(), 
                       src_equeue.size() * sizeof(vid_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(dst + peel_edges, dst_equeue.device_input_ptr(), 
                       dst_equeue.size() * sizeof(vid_t), cudaMemcpyDeviceToHost);

            for (uint32_t i = 0; i < src_equeue.size(); i++) {
                peel[peel_edges + i] = max_peel;
            }

            #if 0
            for (uint32_t i = 0; i < src_equeue.size(); i++) {
                tot_src_equeue.insert(src[peel_edges + i]);
                tot_dst_equeue.insert(dst[peel_edges + i]);
            }
            #endif

            peel_edges += src_equeue.size();
        }

        // remove_bidirect_batch(hornet, src_equeue, dst_equeue);
        
        #if 0
        if (iter_count % 2) {
            oper_bidirect_batch(hornet, src_equeue, dst_equeue, DELETE);
        } else {
            oper_bidirect_batch(h_copy, src_equeue, dst_equeue, DELETE);
        }
        #endif


        src_equeue.clear();
        dst_equeue.clear();
        iter_count++;
        // h_copy.~Hornet();
    }

    json_dump(src, dst, peel, peel_edges);
}


void KCore::release() {
    std::cout << "ran3" << std::endl;
}
}
