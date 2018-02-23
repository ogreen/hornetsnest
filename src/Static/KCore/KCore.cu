#include "Static/KCore/KCore.cuh"
// #include <Device/Primitives/CubWrapper.cuh>

namespace hornets_nest {

KCore::KCore(HornetGraph &hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        src_equeue(hornet),
                        dst_equeue(hornet),
                        tot_src_equeue(hornet),
                        tot_dst_equeue(hornet),
                        load_balancing(hornet) {
                        // nodes_removed(hornet) {
}

KCore::~KCore() {
}

struct CheckDeg {
    TwoLevelQueue<vid_t> vqueue;
    uint32_t peel;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (v.degree() > 0 && v.degree() <= peel) {
            vqueue.insert(id);
        }
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

struct PrintVertices {
    const vid_t *src_ptr;
    const vid_t *dst_ptr;
    int32_t size;

    OPERATOR(Vertex &v) {
        if (v.id() == 0) {
            for (uint32_t i = 0; i < size; i++) {
                printf("%d %d\n", src_ptr[i], dst_ptr[i]);
            }
        }
    }
};

void KCore::reset() {
    std::cout << "ran1" << std::endl;
}

void remove_bidirect_batch(HornetGraph &hornet,
                           TwoLevelQueue<vid_t> src_equeue,
                           TwoLevelQueue<vid_t> dst_equeue) {



    hornet.print();
    std::cout << "\n\n";
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

    std::cout << "sorted by dst\n";
    forAllVertices(hornet, PrintVertices { src_equeue.device_input_ptr(),
                                           dst_equeue.device_input_ptr(),
                                           src_equeue.size() } );
    std::cout << "\n\n";

    hornet.allocateEdgeDeletion(src_equeue.size(), 
                                gpu::batch_property::IN_PLACE);

    // Delete edges in reverse direction.
    hornet.deleteEdgeBatch(batch_update_dst);

    std::cout << "h_copy_mid:\n";
    hornet.print();
    std::cout << "\n\n";



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
    std::cout << "sorted by src\n";
    forAllVertices(hornet, PrintVertices { src_equeue.device_input_ptr(),
                                           dst_equeue.device_input_ptr(),
                                           src_equeue.size() } );
    std::cout << "\n\n";
    #endif

    hornet.allocateEdgeDeletion(src_equeue.size(), 
                                gpu::batch_property::IN_PLACE);

    // Delete edges in the forward direction.
    hornet.deleteEdgeBatch(batch_update_src);
}

void kcores(HornetGraph &hornet, 
            TwoLevelQueue<vid_t> vqueue, 
            TwoLevelQueue<vid_t> src_equeue,
            TwoLevelQueue<vid_t> dst_equeue,
            load_balancing::VertexBased1 load_balancing,
            uint32_t *max_peel,
            uint32_t *ne) {

    uint32_t peel = 0;

    while (*ne > 0) {
        std::cout << "peel: " << peel << "\n";
        forAllVertices(hornet, CheckDeg { vqueue, peel });
        
        vqueue.swap();
        vqueue.print();

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

            remove_bidirect_batch(hornet, src_equeue, dst_equeue);

            *ne -= 2 * src_equeue.size();

            // Save vqueue if ne == 0 -- these are vertices in the kcore.
            if (ne > 0) {
                vqueue.clear();
            }
        } else {
            peel++;    
        }
    }
    *max_peel = peel;
}

HornetGraph* hornet_copy(HornetGraph &hornet, 
                         TwoLevelQueue<vid_t> tot_src_equeue,
                         TwoLevelQueue<vid_t> tot_dst_equeue) {

    HornetInit hornet_init(hornet.nV(), hornet.nE(), hornet.csr_offsets(),
                           hornet.csr_edges(), false);

    auto weights = new int[hornet.nE()]();
    hornet_init.insertEdgeData(weights);

    HornetGraph *h_copy = new HornetGraph(hornet_init);

    return h_copy;
}

void KCore::run() {
    vid_t *src     = new vid_t[hornet.nE() / 2];
    vid_t *dst     = new vid_t[hornet.nE() / 2];
    uint32_t *peel = new uint32_t[hornet.nE() / 2];
    uint32_t peel_edges = 0;
    uint32_t ne = hornet.nE();

    // for (uint32_t i = 0; i < 3; i++) {
    while (peel_edges < hornet.nE() / 2) {
        uint32_t max_peel = 0;
        ne = hornet.nE() - 2 * peel_edges;

        HornetGraph &h_copy = *hornet_copy(hornet, 
                                           tot_src_equeue, 
                                           tot_dst_equeue);

        #if 0
        std::cout << "h_copy_before:\n";
        // hornet.print();
        h_copy.print();
        std::cout << "\n\n";
        #endif
        if (tot_src_equeue.size() > 0) {
            remove_bidirect_batch(h_copy, tot_src_equeue, tot_dst_equeue);
        }

        #if 0
        std::cout << "h_copy_after:\n";
        h_copy.print();
        std::cout << "\n\n";
        #endif

        kcores(h_copy, vqueue, src_equeue, dst_equeue, load_balancing, 
               &max_peel, &ne);

        vqueue.print();
        
        src_equeue.print();
        dst_equeue.print();

        cudaMemcpy(src + peel_edges, src_equeue.device_input_ptr(), 
                   src_equeue.size() * sizeof(vid_t), cudaMemcpyDeviceToHost);

        cudaMemcpy(dst + peel_edges, dst_equeue.device_input_ptr(), 
                   dst_equeue.size() * sizeof(vid_t), cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i < src_equeue.size(); i++) {
            peel[peel_edges + i] = max_peel;
        }

        for (uint32_t i = 0; i < src_equeue.size(); i++) {
            std::cout << src[peel_edges + i] << " " << dst[peel_edges + i] << "\n";
            tot_src_equeue.insert(src[peel_edges + i]);
            tot_dst_equeue.insert(dst[peel_edges + i]);
        }
        peel_edges += src_equeue.size();

        remove_bidirect_batch(hornet, src_equeue, dst_equeue);
        #if 0
        hornet.print();
        std::cout << "\n";
        #endif

        src_equeue.clear();
        dst_equeue.clear();

        h_copy.~Hornet();
    }

    std::cout << "peels:" << "\n";
    for (uint32_t i = 0; i < peel_edges; i++) {
        std::cout << src[i] << " " << dst[i] << " " << peel[i] << "\n";
    }
}

void KCore::release() {
    std::cout << "ran3" << std::endl;
}

}
