#ifndef SNAPSHOTWRITER_H
#define SNAPSHOTWRITER_H

#include <string>
#include <H5Cpp.h>


namespace icy {class SnapshotWriter; class Model;}


class icy::SnapshotWriter
{
public:
    SnapshotWriter();
    icy::Model *model;
    void SaveSnapshot(std::string fileName);
};

#endif // SNAPSHOTWRITER_H
