#include "Static/KCore/KCore.cuh"
#include <Device/Util/Timer.cuh>
#include <Graph/GraphStd.hpp>
#include <Core/Queue/TwoLevelQueue.cuh>

using namespace timer;
using namespace hornets_nest;

int main(int argc, char **argv) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges(), true);

    //auto weights = new int[graph.nE()]();
    //hornet_init.insertEdgeData(weights);

    HornetGraph hornet_graph(hornet_init);

    KCore kcore(hornet_graph);

    kcore.run();

    //delete[] weights;
}
