#include "parameters_sim.h"



void icy::SimParams::Reset()
{
    grid_array = nullptr;
    pts_array = nullptr;

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

    // Snow
    XiSnow = 10.;
    THT_C_snow = 2.0e-3;				// Critical compression
    THT_S_snow = 6.0e-4;				// Critical stretch

    // Drucker-Prager
    // H0 > H3 >=0;
    // H1, H2 >=0
    H0 = 54 * pi / 180.0f;
    H1 = 30 * pi / 180.0f;
    H2 = 0.1f;
    H3 = 40 * pi / 180.0f;

    SandYM = 5e7;

    NACC_xi = 10;
    NACC_max_strain = 0.01;
    IceCompressiveStrength = 100e6;
    IceTensileStrength = 1e6;
    IceShearStrength = 0.5e6;

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;


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
    if(doc.HasMember("NACC_max_strain")) NACC_max_strain = doc["NACC_max_strain"].GetDouble();

    // to be removed
    if(doc.HasMember("NACC_xi")) NACC_xi = doc["NACC_xi"].GetDouble();
    if(doc.HasMember("XiSnow")) XiSnow = doc["XiSnow"].GetDouble();
    if(doc.HasMember("THT_C_snow")) THT_C_snow = doc["THT_C_snow"].GetDouble();
    if(doc.HasMember("THT_S_snow")) THT_S_snow = doc["THT_S_snow"].GetDouble();
    if(doc.HasMember("H0")) H0 = doc["H0"].GetDouble() * pi/180.0;
    if(doc.HasMember("H1")) H1 = doc["H1"].GetDouble() * pi/180.0;
    if(doc.HasMember("H2")) H2 = doc["H2"].GetDouble();
    if(doc.HasMember("H3")) H3 = doc["H3"].GetDouble() * pi/180.0;

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
    real NACC_M = (2*q*sqrt(1+2*beta))/(p0*(1+beta));
    this->NACC_M_sq = NACC_M*NACC_M;
    this->NACC_alpha = std::log(0.991);
}
