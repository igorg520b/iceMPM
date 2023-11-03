#include "snapshotwriter.h"
#include "model.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>


icy::SnapshotWriter::SnapshotWriter()
{

}

void icy::SnapshotWriter::SaveSnapshot(std::string fileName)
{
    spdlog::info("writing snapshot {}",fileName);
    hsize_t nPts = model->prms.nPtsPitch/sizeof(real);

//    H5::IntType datatype_int(H5::PredType::NATIVE_INT);
//    H5::FloatType datatype_double(H5::PredType::NATIVE_DOUBLE);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_params[1] = {sizeof(icy::SimParams)};
    H5::DataSpace dataspace_params(1,dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    hsize_t dims_points[2] = {nPts*icy::SimParams::nPtsArrays,1};
    H5::DataSpace dataspace_points(2, dims_points);
    H5::DSetCreatPropList cparms_points;
    hsize_t chunk_dims[2] = {64000, 1};
    cparms_points.setChunk(2, chunk_dims);
    cparms_points.setDeflate(9);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE,
                                                    dataspace_points, cparms_points);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

    file.close();
    spdlog::info("SaveSnapshot done {}", fileName);
}

void icy::SnapshotWriter::ReadSnapshot(std::string fileName)
{
    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet dataset_nodes = file.openDataSet("Points");
    hsize_t dims_nodes[2];
    dataset_nodes.getSpace().getSimpleExtentDims(dims_nodes, NULL);
    int nNodes = dims_nodes[0]/3;
    std::vector<double> buffer(dims_nodes[0]);
    dataset_nodes.read(buffer.data(),H5::PredType::NATIVE_DOUBLE);

    model->points.resize(nNodes);
    for(int i=0;i<nNodes;i++)
    {
        icy::Point &p = model->points[i];
        p.pos[0] = buffer[i];
        p.pos[1] = buffer[i+nNodes];
        p.NACC_alpha_p = buffer[i+nNodes*2];
    }
    file.close();
}
