#ifndef SNAPSHOTWRITER_H
#define SNAPSHOTWRITER_H

#include <string>


namespace icy {class SnapshotWriter; class Model;}


class icy::SnapshotWriter
{
public:
    icy::Model *model;
    void SaveSnapshot(std::string fileName, bool fullData);
    void ReadSnapshot(std::string fileName);
};

#endif // SNAPSHOTWRITER_H
