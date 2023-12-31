#include "parameters_sim.h"



void icy::SimParams::Reset()
{
    grid_array = nullptr;
    pts_array = nullptr;
    indenter_force_accumulator = nullptr;

    InitialTimeStep = 3.e-5;
    YoungsModulus = 5.e8;
    PointsWanted = 50'000;
    GridX = 128;
    GridY = 55;
    ParticleViewSize = 2.5f;
    GridXDimension = 3.33;

    SimulationEndTime = 12;

    PoissonsRatio = 0.3;
    Gravity = 9.81;
    Density = 980;
    IceFrictionCoefficient = 0.03;

    IndDiameter = 0.324;
    IndVelocity = 0.2;
    IndDepth = 0.25;//0.101;

    BlockHeight = 1.0;
    BlockLength = 2.5;
    HoldBlockOnTheRight = 0;

    SimulationStep = 0;
    SimulationTime = 0;

    IceCompressiveStrength = 100e6;
    IceTensileStrength = 1e6;
    IceShearStrength = 0.5e6;

    DP_tan_phi = std::tan(30*pi/180.);

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;

    indenter_x = indenter_x_initial = indenter_y = 0;

    ComputeLame();
    ComputeCamClayParams2();
    ComputeHelperVariables();
    std::cout << "SimParams Reset() done\n";
}


std::string icy::SimParams::ParseFile(std::string fileName)
{
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("configuration file is not found");
    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();

    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject()) throw std::runtime_error("configuration file is not JSON");

    std::string outputDirectory = "output";
    if(doc.HasMember("OutputDirectory")) outputDirectory = doc["OutputDirectory"].GetString();
    if(doc.HasMember("InitialTimeStep")) InitialTimeStep = doc["InitialTimeStep"].GetDouble();
    if(doc.HasMember("YoungsModulus")) YoungsModulus = doc["YoungsModulus"].GetDouble();
    if(doc.HasMember("PointsWanted")) PointsWanted = doc["PointsWanted"].GetDouble();
    if(doc.HasMember("GridX")) GridX = doc["GridX"].GetInt();
    if(doc.HasMember("GridY")) GridY = doc["GridY"].GetInt();
    if(doc.HasMember("GridXDimension")) GridXDimension = doc["GridXDimension"].GetDouble();
    if(doc.HasMember("HoldBlockOnTheRight")) HoldBlockOnTheRight = doc["HoldBlockOnTheRight"].GetInt();
    if(doc.HasMember("ParticleViewSize")) ParticleViewSize = doc["ParticleViewSize"].GetDouble();
    if(doc.HasMember("SimulationEndTime")) SimulationEndTime = doc["SimulationEndTime"].GetDouble();
    if(doc.HasMember("PoissonsRatio")) PoissonsRatio = doc["PoissonsRatio"].GetDouble();
    if(doc.HasMember("Gravity")) Gravity = doc["Gravity"].GetDouble();
    if(doc.HasMember("Density")) Density = doc["Density"].GetDouble();
    if(doc.HasMember("IceFrictionCoefficient")) IceFrictionCoefficient = doc["IceFrictionCoefficient"].GetDouble();
    if(doc.HasMember("IndDiameter")) IndDiameter = doc["IndDiameter"].GetDouble();
    if(doc.HasMember("IndVelocity")) IndVelocity = doc["IndVelocity"].GetDouble();
    if(doc.HasMember("IndDepth")) IndDepth = doc["IndDepth"].GetDouble();
    if(doc.HasMember("BlockHeight")) BlockHeight = doc["BlockHeight"].GetDouble();
    if(doc.HasMember("BlockLength")) BlockLength = doc["BlockLength"].GetDouble();

    if(doc.HasMember("IceCompressiveStrength")) IceCompressiveStrength = doc["IceCompressiveStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength")) IceTensileStrength = doc["IceTensileStrength"].GetDouble();
    if(doc.HasMember("IceShearStrength")) IceShearStrength = doc["IceShearStrength"].GetDouble();

    if(doc.HasMember("DP_phi")) DP_tan_phi = std::tan(doc["DP_phi"].GetDouble()*pi/180);


    ComputeCamClayParams2();
    ComputeLame();
    ComputeHelperVariables();

    std::cout << "loaded parameters file " << fileName << '\n';
    std::cout << "GridXDimension " << GridXDimension << '\n';
    std::cout << "cellsize " << cellsize << '\n';
    return outputDirectory;
}

void icy::SimParams::ComputeLame()
{
    lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
    mu = YoungsModulus/(2*(1+PoissonsRatio));
    kappa = mu*2./3. + lambda;
}

void icy::SimParams::ComputeHelperVariables()
{
    UpdateEveryNthStep = (int)(1.f/(200*InitialTimeStep));
//    UpdateEveryNthStep = (int)(1.f/(400*InitialTimeStep));
    cellsize = GridXDimension/GridX;
    cellsize_inv = 1./cellsize;
    Dp_inv = 4./(cellsize*cellsize);
    IndRSq = IndDiameter*IndDiameter/4.;
}

void icy::SimParams::ComputeCamClayParams2()
{
    ComputeLame();
    NACC_beta = IceTensileStrength/IceCompressiveStrength;
    const real &beta = NACC_beta;
    const real &q = IceShearStrength;
    const real &p0 = IceCompressiveStrength;
    NACC_M = (2*q*sqrt(1+2*beta))/(p0*(1+beta));
    NACC_Msq = NACC_M*NACC_M;
}
