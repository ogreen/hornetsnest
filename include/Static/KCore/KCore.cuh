#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

namespace hornets_nest {

using HornetGraph = gpu::Csr<EMPTY, EMPTY>;

class KCore : public StaticAlgorithm<HornetGraph> {
public:
    KCore(HornetGraph& hornet);
    ~KCore();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

private:
    // HostDeviceVar<KCoreData> hd_data;

    //load_balancing::BinarySearch load_balancing;
    //load_balancing::VertexBased1 load_balancing;

    TwoLevelQueue<vid_t> vqueue;
    TwoLevelQueue<vid_t> equeue;
    // MultiLevelQueue<vid_t> nodes_removed;
};

}
