/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BetweennessCentrality/bc.cuh"
#include "Static/BetweennessCentrality/exact_bc.cuh"
#include "Static/BetweennessCentrality/approximate_bc.cuh"
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off


#include <vector>
 
using namespace std;
using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;



int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    // GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);
    Timer<DEVICE> TM;

    int numGPUs=2;

    if(false){

        HornetGraph** hornets = new HornetGraph*[numGPUs];
        // std::vector<HornetGraph> hornets(numGPUs);

        // Create a single Hornet Graph for each GPU
        #pragma omp parallel for
            for (int thread_id=0; thread_id < numGPUs; thread_id++){
                cudaSetDevice(thread_id);

                HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                                       graph.csr_out_edges());
                // hornets[thread_id] = HornetGraph(hornet_init);

                hornets[thread_id] = new HornetGraph(hornet_init);
            }


        ApproximateBC** BCs = new ApproximateBC*[numGPUs];

        #pragma omp parallel for
            for (int thread_id=0; thread_id < numGPUs; thread_id++){

                cudaSetDevice(thread_id);


                vid_t* roots = new vid_t[graph.nV()/numGPUs+1];

                int i=0;
                for(int v=thread_id; v<graph.nV(); v+=numGPUs){
                    roots[i++]=v;
                }
                // cout << endl;

                BCs[thread_id] = new ApproximateBC(*hornets[thread_id],roots,i);
                // BCs[thread_id] = new ApproximateBC(hornets[thread_id],roots,i);
                BCs[thread_id]->reset();
                delete[] roots;
            }

        #pragma omp parallel for
            for (int thread_id=0; thread_id < numGPUs; thread_id++){
                cudaSetDevice(thread_id);

                BCs[thread_id]->run();
            }


        // Destroying all the Hornet graphs
        #pragma omp parallel for
            for (int thread_id=0; thread_id < numGPUs; thread_id++){
                delete BCs[thread_id];
                delete hornets[thread_id];
            }
        delete [] BCs;
        delete [] hornets;

    }
    else{


        // Create a single Hornet Graph for each GPU
        // #pragma omp parallel for
        //     for (int thread_id=0; thread_id < numGPUs; thread_id++){
        #pragma omp parallel
        {      
            int32_t thread_id = omp_get_thread_num ();
            cudaSetDevice(thread_id);
            HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                                   graph.csr_out_edges());
            HornetGraph hornet_graph(hornet_init);

            vid_t* roots = new vid_t[graph.nV()/numGPUs+1];

            int i=0;
            for(int v=thread_id; v<graph.nV(); v+=numGPUs){
                roots[i++]=v;
            }


            ApproximateBC bc(hornet_graph,roots,i);
            bc.reset();
            delete[] roots;

            bc.run();
        }


    }


}
