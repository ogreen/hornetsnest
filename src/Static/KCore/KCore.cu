#include "Static/KCore/KCore.cuh"

namespace hornets_nest {

KCore::KCore(HornetGraph& hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        src_equeue(hornet),
                        dst_equeue(hornet),
                        load_balancing(hornet) {
                        // nodes_removed(hornet) {
}

KCore::~KCore() {
}

struct CheckDeg {
    TwoLevelQueue<vid_t> vqueue;
    int peel;

    OPERATOR(Vertex& v) {
        vid_t id = v.id();
        if (v.degree() <= peel) {
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

void KCore::reset() {
    std::cout << "ran1" << std::endl;
}

void KCore::run() {
    //cuMalloc(hd_data().src, hornet.nE());
    //cuMalloc(hd_data().dst, hornet.nE());
    //hd_data().counter = 0;

    int peel = 2;
    bool changed = true;
    while (hornet.nE() > 0) {
        if (changed) {
            changed = false;
            forAllVertices(hornet, CheckDeg { vqueue, peel });
            
            vqueue.swap();
            vqueue.print();

            if (vqueue.size() > 0) {
                changed = true;
                forAllEdges(hornet, vqueue, 
                             PeelVertices { src_equeue, dst_equeue }, 
                            load_balancing); 

                src_equeue.swap();
                dst_equeue.swap();

                BatchUpdate batch_update((vid_t*) src_equeue.device_input_ptr(),
                                         (vid_t*) dst_equeue.device_input_ptr(),
                                         src_equeue.size(), 
                                         gpu::BatchType::DEVICE);

                hornet.deleteEdgeBatch(batch_update);

                printf("ne: %d\n", hornet.nE());
            }

            vqueue.clear();
            break;

        }// else {
            
        //}
    }
}

void KCore::release() {
    std::cout << "ran3" << std::endl;
}

}
