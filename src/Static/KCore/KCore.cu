#include "Static/KCore/KCore.cuh"

namespace hornets_nest {

KCore::KCore(HornetGraph& hornet) : 
                        StaticAlgorithm(hornet),
                        vqueue(hornet),
                        equeue(hornet) {
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
    TwoLevelQueue<vid_t> equeue;

    OPERATOR(Vertex &v, Edge &e) {
        auto dst = e.dst_id();
        equeue.insert(dst);
    }
};

void KCore::reset() {
    std::cout << "ran1" << std::endl;
}

void KCore::run() {
    int peel = 2;
    bool changed = true;
    while (hornet.nE() > 0) {
        if (changed) {
            changed = false;
            forAllVertices(hornet, CheckDeg { vqueue, peel });
            
            vqueue.swap();
            vqueue.print();
            break;

            #if 0
            if (vqueue.size() > 0) {
                changed = true;
                forAllEdges(hornet, vqueue, PeelVertices{ equeue }); 
            }

            vqueue.clear();
            equeue.clear();
            #endif

        }// else {
            
        //}
    }
}

void KCore::release() {
    std::cout << "ran3" << std::endl;
}

}
