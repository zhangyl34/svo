#include <svo/config.h>
#include <svo/global.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>
#include <vikit/camera_loader.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.h>

#include "thirdParty/DBoW2/DUtils/Random.h"



namespace svo {
namespace initialization {

FramePtr frame_ref_;          // 保存了第一帧的图像信息
vector<cv::Point2f> px_ref_;  // 第一帧特征点
vector<cv::Point2f> px_cur_;  // 第二帧特征点
vector<Vector3d> f_ref_;      // 第一帧特征点
vector<Vector3d> f_cur_;      // 第二帧特征点

// F 矩阵所需参数
float mSigma = 1.0;

// FAST 角点检测
void detectFeatures(FramePtr frame, vector<cv::Point2f>& px_vec, vector<Vector3d>& f_vec) {
    Features new_features;
    // Fast 角点检测
    feature_detection::FastDetector detector(
        frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());  // ,, 30, 3
    detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);  // ,, 20,

    // now for all maximum corners, initialize a new seed
    px_vec.clear(); px_vec.reserve(new_features.size());
    f_vec.clear(); f_vec.reserve(new_features.size());
    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
        px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));  // 像素坐标
        f_vec.push_back(ftr->f);
        delete ftr;
    });
}

// 光流法追踪
void trackKlt(FramePtr frame_ref, FramePtr frame_cur, vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur, vector<Vector3d>& f_ref, vector<Vector3d>& f_cur,
    vector<double>& disparities) {

    const double klt_win_size = 30.0;
    const int klt_max_iter = 30;
    const double klt_eps = 0.001;
    vector<uchar> status;
    vector<float> error;
    vector<float> min_eig_vec;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
    cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                            px_ref, px_cur,
                            status, error,
                            cv::Size2i(klt_win_size, klt_win_size),
                            4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
    vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
    vector<Vector3d>::iterator f_ref_it = f_ref.begin();
    f_cur.clear(); f_cur.reserve(px_cur.size());
    disparities.clear(); disparities.reserve(px_cur.size());
    for(size_t i=0; px_ref_it != px_ref.end(); ++i) {
        if(!status[i]) {  // 没追踪到的特征点，直接永久性移除
            px_ref_it = px_ref.erase(px_ref_it);
            px_cur_it = px_cur.erase(px_cur_it);
            f_ref_it = f_ref.erase(f_ref_it);
            continue;
        }
        f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
        disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
        ++px_ref_it;
        ++px_cur_it;
        ++f_ref_it;
    }
}

// 计算 H 矩阵
void computeHomography(const vector<Vector3d>& f_ref, const vector<Vector3d>& f_cur,
    double focal_length, double reprojection_threshold, vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur, SE3& T_cur_from_ref) {
  vector<Vector2d > uv_ref(f_ref.size());
  vector<Vector2d > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);  // uv 并非像素坐标，只是取了 f 的前两位
    uv_cur[i] = vk::project2d(f_cur[i]);
  }
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();
  vector<int> outliers;
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}

// 计算 F 矩阵：正则化 https://www.cnblogs.com/hardjet/p/11460822.html
void Normalize(const vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) {
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();
    vNormalizedPoints.resize(N);
    for(int i=0; i<N; i++) {
        meanX += vKeys[i].x;
        meanY += vKeys[i].y;
    }
    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;
    for(int i=0; i<N; i++) {
        vNormalizedPoints[i].x = vKeys[i].x - meanX;
        vNormalizedPoints[i].y = vKeys[i].y - meanY;
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++) {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

// 计算 F 矩阵：八点计算 F
cv::Mat ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2) {
    const int N = vP1.size();  // 8
    cv::Mat A(N,9,CV_32F);
    for(int i=0; i<N; i++) {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat Fpre = vt.row(8).reshape(0, 3);  // 最后一行，对应特征值 0
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

// 计算 F 矩阵：检查内点
float CheckFundamental(const vector<cv::Point2f>& px_ref, const vector<cv::Point2f>& px_cur,
    const cv::Mat &F21, vector<bool> &vbMatchesInliers) {
    
    const int N = px_ref.size();
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);
    vbMatchesInliers.resize(N);
    float score = 0;
    const float th = 3.841;       // inlier 阈值
    const float thScore = 5.991;  // 分数
    const float invSigmaSquare = 1.0/(mSigma*mSigma);

    for(int i=0; i<N; i++) {
        bool bIn = true;
        const cv::Point2f &kp1 = px_ref[i];
        const cv::Point2f &kp2 = px_cur[i];
        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);  // 特征点到极线的距离
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);  // 特征点到极线的距离
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;          

        if(bIn) {
            score += thScore - chiSquare1;
            score += thScore - chiSquare2;
            vbMatchesInliers[i]=true;
        }
        else {
            vbMatchesInliers[i]=false;
        }
    }

    return score;
}

// 计算 F 矩阵
void FindFundamental(const vector<cv::Point2f>& px_ref, const vector<cv::Point2f>& px_cur,
    vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21) {

    // 正则化
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(px_ref, vPn1, T1);
    Normalize(px_cur, vPn2, T2);
    cv::Mat T2t = T2.t();

    const int mMaxIterations = 200;
    const int N = vPn1.size();
    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);
    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Indices for minimum set selection
    vector<size_t> vAllIndices;  // 0 到 N-1
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;
    for(int i=0; i<N; i++) {
        vAllIndices.push_back(i);
    }
    
    // Generate sets of 8 points for each RANSAC iteration
    vector<vector<size_t>> mvSets;  // 放的是 id
    mvSets = vector<vector<size_t>>(mMaxIterations,vector<size_t>(8,0));
    DUtils::Random::SeedRandOnce(0);
    for(int it=0; it<mMaxIterations; it++) {
        vAvailableIndices = vAllIndices;
        // Select a minimum set
        for(size_t j=0; j<8; j++) {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            size_t idx = vAvailableIndices[randi];
            mvSets[it][j] = idx;
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++) {
        // Select a minimum set
        for(int j=0; j<8; j++) {
            int idx = mvSets[it][j];
            vPn1i[j] = vPn1[idx];
            vPn2i[j] = vPn2[idx];
        }
        cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(px_ref, px_cur, F21i, vbCurrentInliers);

        if(currentScore>score) {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
    
    // int inliers_number = 0;
    // for(int i=0; i<vbMatchesInliers.size(); i++) {
    //     if (vbMatchesInliers[i]) {
    //         inliers_number ++;
    //     }
    // }
    // SVO_INFO_STREAM("my score: "<<std::to_string(currentScore)<<"; my inliers: "<<std::to_string(inliers_number));
    // float tempScore;
    // cv::Mat tempF(3, 3, CV_32F);
    // tempF.at<float>(0,0) = 1.195619996560168e-06;
    // tempF.at<float>(0,1) = 1.5830418562518e-05;
    // tempF.at<float>(0,2) = -0.01257347941573252;
    // tempF.at<float>(1,0) = -1.281177574115178e-05;
    // tempF.at<float>(1,1) = -9.205286870749917e-06;
    // tempF.at<float>(1,2) = 0.02440392860727506;
    // tempF.at<float>(2,0) = 0.01097453773243918;
    // tempF.at<float>(2,1) = -0.0239656030179479;
    // tempF.at<float>(2,2) = 1;
    // vector<bool> tempInliers(N,false);
    // tempScore = CheckFundamental(px_ref, px_cur, tempF, tempInliers);
    // inliers_number = 0;
    // for(int i=0; i<tempInliers.size(); i++) {
    //     if (tempInliers[i]) {
    //         inliers_number ++;
    //     }
    // }
    // SVO_INFO_STREAM("cv score: "<<std::to_string(tempScore)<<"; cv inliers: "<<std::to_string(inliers_number));
    // F21 = tempF.clone();
    // vbMatchesInliers = tempInliers;
    // score = tempScore;
}

// 分解 F 矩阵：分解 E
void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
    cv::Mat u,w,vt;
    cv::SVDecomp(E,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // SVO_INFO_STREAM("DecomposeE: u: "<<u<<"w: "<<w<<"vt: "<<vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

// 分解 F 矩阵：三角化
void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1,
    const cv::Mat &P2, cv::Mat &x3D) {

    cv::Mat A(4,4,CV_32F);

    cout << "p1:" << kp1 << "p2:" << kp2 << endl;

    A.row(0) = kp1.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
    cout << "x3D:" << x3D << endl;
}

// 分解 F 矩阵：检查
int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f>& px_ref,
    const vector<cv::Point2f>& px_cur, const vector<bool> &vbMatchesInliers, const cv::Mat &K,
    vector<cv::Point3f> &vP3D, const float th2, vector<bool> &vbGood, float &parallax) {

    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);
    vbGood = vector<bool>(px_ref.size(),false);
    vP3D.resize(px_ref.size());  // 113

    vector<float> vCosParallax;
    vCosParallax.reserve(px_ref.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    // cout << "P1: " << P1 << endl;

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;
    // cout << "P2: " << P2 << endl;

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);  // 相机空间坐标
    cv::Mat O2 = -R.t()*t;
    int nGood=0;
    for(size_t i=0, iend=px_ref.size();i<iend;i++) {
        if(!vbMatchesInliers[i])
            continue;

        const cv::Point2f &kp1 = px_ref[i];
        const cv::Point2f &kp2 = px_cur[i];
        cv::Mat p3dC1;
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            SVO_INFO_STREAM("infinite test fail.");
            vbGood[i]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);
        // 深度 >0
        if(p3dC1.at<float>(2)<=0) {  // && cosParallax>0.99998
            SVO_INFO_STREAM("depth test1 fail.");
            continue;
        }
        cv::Mat p3dC2 = R*p3dC1+t;
        if(p3dC2.at<float>(2)<=0) {  // && cosParallax<0.99998
            SVO_INFO_STREAM("depth test2 fail.");
            continue;
        }

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        // cout << "fx: " << std::to_string(fx) << "cx: " << std::to_string(cx) << endl;
        // cout << "fy: " << std::to_string(fy) << "cy: " << std::to_string(cy) << endl;
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
        // cout << "im1x: " << std::to_string(im1x) << "im1y: " << std::to_string(im1y) << endl;
        float squareError1 = (im1x-kp1.x)*(im1x-kp1.x)+(im1y-kp1.y)*(im1y-kp1.y);  // 7*7+7*7=98
        SVO_INFO_STREAM("squareError1: "<<std::to_string(squareError1));
        if(squareError1>th2)  // >4.0
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;
        float squareError2 = (im2x-kp2.x)*(im2x-kp2.x)+(im2y-kp2.y)*(im2y-kp2.y);
        SVO_INFO_STREAM("squareError2: "<<std::to_string(squareError2));
        if(squareError2>th2)  // >4.0
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[i] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        vbGood[i]=true;
        // if(cosParallax<0.99998)
        //     vbGood[i]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

// 分解 F 矩阵
bool ReconstructF(const vector<cv::Point2f>& px_ref, const vector<cv::Point2f>& px_cur, 
    const vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K, cv::Mat &R21,
    cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax,
    int minTriangulated) {

    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size(); i<iend; i++)
        if(vbMatchesInliers[i])
            N++;  // 106

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;
    cv::Mat R1, R2, t;
    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);
    cout << "R1: " << R1 << "; R2: " << R2 << endl;
    cv::Mat t1=t;
    cv::Mat t2=-t;
    cout << "t: " << t << endl;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4, vP3D5;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3,vbTriangulated4,vbTriangulated5;
    float parallax1,parallax2,parallax3,parallax4,parallax5;
    float mSigma2 = mSigma*mSigma;
    // int nGood1 = CheckRT(R1,t1,px_ref,px_cur,vbMatchesInliers,K,vP3D1,4.0*mSigma2,vbTriangulated1,parallax1);
    // int nGood2 = CheckRT(R2,t1,px_ref,px_cur,vbMatchesInliers,K,vP3D2,4.0*mSigma2,vbTriangulated2,parallax2);
    // int nGood3 = CheckRT(R1,t2,px_ref,px_cur,vbMatchesInliers,K,vP3D3,4.0*mSigma2,vbTriangulated3,parallax3);
    // int nGood4 = CheckRT(R2,t2,px_ref,px_cur,vbMatchesInliers,K,vP3D4,4.0*mSigma2,vbTriangulated4,parallax4);
    int nGood1, nGood2, nGood3, nGood4;
    // cv::Mat R5(3, 3, CV_32F);
    // R5.at<float>(0,0) = 0.9954;
    // R5.at<float>(0,1) = -0.0563;
    // R5.at<float>(0,2) = 0.0774;
    // R5.at<float>(1,0) = 0.0586;
    // R5.at<float>(1,1) = 0.9979;
    // R5.at<float>(1,2) = -0.0272;
    // R5.at<float>(2,0) = -0.0757;
    // R5.at<float>(2,1) = 0.0316;   
    // R5.at<float>(2,2) = 0.9966;
    // cv::Mat t5(3, 1, CV_32F);
    // t5.at<float>(0,0) = 3.1400;
    // t5.at<float>(0,1) = -1.0651;
    // t5.at<float>(0,2) = -0.1819;
    // cv::Mat R5(3, 3, CV_32F);
    // R5.at<float>(0,0) = 0.9954;
    // R5.at<float>(0,1) = -0.0563;
    // R5.at<float>(0,2) = 0.0774;
    // R5.at<float>(1,0) = 0.0586;
    // R5.at<float>(1,1) = 0.9979;
    // R5.at<float>(1,2) = -0.0272;
    // R5.at<float>(2,0) = -0.0757;
    // R5.at<float>(2,1) = 0.0316;   
    // R5.at<float>(2,2) = 0.9966;
    // cv::Mat t5(3, 1, CV_32F);
    // t5.at<float>(0,0) = -0.4450148;
    // t5.at<float>(0,1) = 0.17372292;
    // t5.at<float>(0,2) = -0.87851131;
    cv::Mat R5(3, 3, CV_32F);
    R5.at<float>(0,0) = 0.99695;
    R5.at<float>(0,1) = -0.0368003;
    R5.at<float>(0,2) = 0.0688202;
    R5.at<float>(1,0) = 0.0373442;
    R5.at<float>(1,1) = 0.99928;
    R5.at<float>(1,2) = -0.0066322;
    R5.at<float>(2,0) = -0.0685266;
    R5.at<float>(2,1) =  0.00918201;   
    R5.at<float>(2,2) = 0.997607;
    cv::Mat t5(3, 1, CV_32F);
    t5.at<float>(0,0) = -0.00119094;
    t5.at<float>(0,1) = 0.00967048;
    t5.at<float>(0,2) = 0.00328677;
    int nGood5 = CheckRT(R5,t5,px_ref,px_cur,vbMatchesInliers,K,vP3D5,4.0*mSigma2,vbTriangulated5,parallax5);
    cout << nGood1 << " " << nGood2 << " " << nGood3 << " " << nGood4 << " " << nGood5 << endl;
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);  // (,50)
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1) {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1) {
        if(parallax1>minParallax) {  // >1.0
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if(maxGood==nGood2) {
        if(parallax2>minParallax) {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if(maxGood==nGood3) {
        if(parallax3>minParallax) {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    else if(maxGood==nGood4) {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    return false;
}



InitResult addFirstFrame(FramePtr frame_ref) {
    // FAST 角点检测，px_ref_ 是特征点的像素坐标，f_ref_ 是特征点在相机坐标系下的坐标（Z 归一化为 1）
    detectFeatures(frame_ref, px_ref_, f_ref_);
    SVO_INFO_STREAM("Found " << px_ref_.size() << " features.");
    frame_ref_ = frame_ref;
    px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());  // 用第一帧初始化第二帧

    // 绘制特征点
    // cv::Mat imgRaw;
    // cv::cvtColor(frame_ref->img_pyr_[0].clone(), imgRaw, cv::COLOR_GRAY2RGB);
    // for(vector<cv::Point2f>::const_iterator it_ref=px_ref_.begin();
    //     it_ref != px_ref_.end(); ++it_ref)
    //     cv::circle(imgRaw, cv::Point2f(it_ref->x,it_ref->y), 2, cv::Scalar(0,255,0), -1);
    // cv::imshow("test", imgRaw);
    // cv::waitKey(0);
    return SUCCESS;
}

InitResult addSecondFrame(FramePtr frame_cur) {
    vector<double> disparities_;  // 光流的距离
    vector<int> inliers_;  // H 的内点
    vector<Vector3d> xyz_in_cur_;
    Sophus::SE3 T_cur_from_ref_;

    // 调用 opencv 函数，光流法追踪特征点
    trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
    SVO_INFO_STREAM("KLT tracked "<< disparities_.size() <<" features");  // 期望 >50
    double disparity = vk::getMedian(disparities_);  // 取中位数
    SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");  // 期望 >50

    // 绘制光流
    // cv::Mat img_lf;
    // cv::cvtColor(frame_cur->img_pyr_[0].clone(), img_lf, cv::COLOR_GRAY2RGB);
    // for(vector<cv::Point2f>::const_iterator it_ref=px_ref_.begin(), it_cur=px_cur_.begin();
    //     it_ref != px_ref_.end(); ++it_ref, ++it_cur)
    //     cv::line(img_lf, cv::Point2f(it_cur->x, it_cur->y), cv::Point2f(it_ref->x, it_ref->y),
    //         cv::Scalar(0,255,0), 2);
    // cv::imshow("line flow", img_lf);
    // cv::waitKey(0);

    // H 矩阵
    // computeHomography(f_ref_, f_cur_, frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
    //     inliers_, xyz_in_cur_, T_cur_from_ref_);  // ,, 578, 2,,,
    // SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");  // 期望 >40
    // SVO_INFO_STREAM("R_cur_ref: "<<T_cur_from_ref_.rotation_matrix()<<'\n'<<"t_cur_ref: "<<T_cur_from_ref_.translation());  // SE3

    // 绘制 H 矩阵的内点
    // cv::Mat img_hi;
    // cv::cvtColor(frame_cur->img_pyr_[0].clone(), img_hi, cv::COLOR_GRAY2RGB);
    // for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it) {
    //     Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    //     Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    //     if(xyz_in_cur_[*it].z() > 0) {
    //         cv::line(img_hi, cv::Point2f(px_cur(0), px_cur(1)), cv::Point2f(px_ref(0), px_ref(1)),
    //             cv::Scalar(255,255,0), 2);
    //     }
    // }
    // cv::imshow("H inliers", img_hi);
    // cv::waitKey(0);

    // F 矩阵
    vector<bool> vbMatchesInliersF;  // 大小与光流相同
    float SF;  // score
    cv::Mat F;
    FindFundamental(px_ref_, px_cur_, vbMatchesInliersF, SF, F);
    // SVO_INFO_STREAM("F: "<<F);
    cv::Mat mK, R21, t21;  // mK 是相机内参
    vk::PinholeCamera* tempCam = dynamic_cast<vk::PinholeCamera*>(frame_ref_->cam_);
    cv::eigen2cv(tempCam->K(), mK);
    mK.convertTo(mK, CV_32F);
    vector<cv::Point3f> vP3D;
    vector<bool> vbTriangulated;
    if (!ReconstructF(px_ref_,px_cur_,vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50)) {
        SVO_INFO_STREAM("Reconstruction fail.");
    }
    else {
        SVO_INFO_STREAM("T_cur_ref: "<<R21<<"; "<<t21);
    }

    // 绘制 F 矩阵的内点
    cv::Mat imgRaw;
    cv::cvtColor(frame_cur->img_pyr_[0].clone(), imgRaw, cv::COLOR_GRAY2RGB);
    int FInliers = 0;
    for(size_t i=0; i<vbMatchesInliersF.size(); ++i) {
        Vector2d px_cur(px_cur_[i].x, px_cur_[i].y);
        Vector2d px_ref(px_ref_[i].x, px_ref_[i].y);
        if(vbMatchesInliersF[i]==true) {  // xyz_in_cur_[i].z()>0 ?
            FInliers ++;
            cv::line(imgRaw, cv::Point2f(px_cur(0), px_cur(1)), cv::Point2f(px_ref(0), px_ref(1)),
                cv::Scalar(255,255,0), 2);
        }
    }
    SVO_INFO_STREAM("Init: Fundamental RANSAC "<<FInliers<<" inliers.");  // 期望 >40
    cv::imshow("test", imgRaw);
    cv::waitKey(0);

    return SUCCESS;
}

}  // namespace initialization
}  // namespace svo

int main(int argc, char *argv[]) {
    vk::AbstractCamera* cam_;
    vk::camera_loader::loadForNeal(cam_);
    cv::Mat img = cv::imread("/home/neal/projects/svo/toolkit/data/rii2_img/1162.png", 0);
    svo::FramePtr first_frame_ ;
    first_frame_.reset(new svo::Frame(cam_, img, double(1.0)));
    first_frame_->T_f_w_ = Sophus::SE3(svo::Matrix3d::Identity(), svo::Vector3d::Zero());
    svo::initialization::addFirstFrame(first_frame_);

    // ./neal_test . 1207
    cv::Mat img2 = cv::imread("/home/neal/projects/svo/toolkit/data/rii2_img/"+std::string(argv[2])+".png", 0);
    svo::FramePtr second_frame_ ;
    second_frame_.reset(new svo::Frame(cam_, img2, double(2.0)));
    svo::initialization::addSecondFrame(second_frame_);

    // cv::Mat F21(3, 3, CV_32F);
    // F21.at<float>(0,0) = 1.195619996560168e-06;
    // F21.at<float>(0,1) = 1.5830418562518e-05;
    // F21.at<float>(0,2) = -0.01257347941573252;
    // F21.at<float>(1,0) = -1.281177574115178e-05;
    // F21.at<float>(1,1) = -9.205286870749917e-06;
    // F21.at<float>(1,2) = 0.02440392860727506;
    // F21.at<float>(2,0) = 0.01097453773243918;
    // F21.at<float>(2,1) = -0.0239656030179479;
    // F21.at<float>(2,2) = 1;
    // cv::Mat mK;
    // vk::PinholeCamera* tempCam = dynamic_cast<vk::PinholeCamera*>(cam_);
    // cv::eigen2cv(tempCam->K(), mK);
    // mK.convertTo(mK, CV_32F);
    // cv::Mat E21 = mK.t()*F21*mK;
    // cv::Mat R1, R2, t;
    // // Recover the 4 motion hypotheses
    // svo::initialization::DecomposeE(E21,R1,R2,t);
    // std::cout << R1 << '\n' << R2 << '\n' << t << std::endl;

    return 0;
}