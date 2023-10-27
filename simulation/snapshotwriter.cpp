#include "snapshotwriter.h"
#include "model.h"
#include <spdlog/spdlog.h>


icy::SnapshotWriter::SnapshotWriter()
{

}

void SaveSnapshot(std::string fileName)
{
    spdlog::info("writing snapshot {}",fileName);

    H5::IntType datatype_int(H5::PredType::NATIVE_INT);
    H5::FloatType datatype_double(H5::PredType::NATIVE_DOUBLE);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

//    hsize_t dimsf_nodes_new[2] = {model, 2};
//    H5::DataSpace dataspace_nodes_new(2, dimsf_nodes_new);
//    H5::DataSet dataset_nodes_new = file.createDataSet("Nodes_New", datatype_double, dataspace_nodes_new);
//    dataset_nodes_new.write(nodes_buffer, H5::PredType::NATIVE_DOUBLE);

    spdlog::info("SaveSnapshot done");
}
