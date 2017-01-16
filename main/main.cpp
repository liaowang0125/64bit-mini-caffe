#include "common/fs.hpp"
#include "common/common.hpp"
#include "caffe/net.hpp"
#include <caffe/caffe.hpp>  
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace common;
using namespace cv;
using namespace caffe;
using namespace fs;

using std::cout;
using std::cin;
using std::endl;
using caffe::Blob;

string imgdir = "../cam2";
float S = 800;
float L = 2;

static Mat img_resize(string &filepath){
	Image image = Image::Open(filepath.c_str());
	Mat img = image.img;
	float bigger = max(img.size[0], img.size[1]);
	float ratio = S / bigger;
	resize(img, img, Size(img.size[1] * ratio,img.size[0]*ratio));
	
	return img;
}
static vector<vector<float>> get_rmac_region_coordinates(const cv::Mat& img){
	float H = img.size[0];
	float W = img.size[1];
	float ovr = 0.4;
	Mat a;
	vector<float> steps{ 1, 2, 3, 4, 5, 6 };
	vector<float> idx;
	float w = min(H, W);
	for (int i = 0; i < steps.size(); i++){
		float a = (max(H, W) - w) / steps[i];
		float b = abs(((w*w - w*a) / (w*w)) - ovr);
		idx.push_back(b);		
	}
	vector<float>::iterator minimum = std::min_element(idx.begin(),idx.end());
	//cout << *minimum << endl;
	int id=distance(std::begin(idx), minimum)+1;
	
	int Wd = 0;
	int Hd = 0;
	if (H < W)Wd = id;
	else if (H>W)Hd = id;
	vector<vector<float>>regions_xywh;
	for (int l = 1; l < L+1; l++){
		float wl = floor(2 * w / (l + 1));
		float wl2 = floor(wl / 2 - 1);
		float b;
		if (l + Wd - 1 > 0){
			b = (W - wl) / (l + Wd - 1);
		}
		else{
			b = 0;
		}
		vector<float>cenW;
		for (int i = 0; i < l - 1 + Wd + 1; i++)
		{
			float tmp = floor(wl2 + b*i) - wl2;
			cenW.push_back(tmp);
		}
		if (l + Hd - 1 > 0){
			b = (H - wl) / (l + Hd - 1);
		}
		else{
			b = 0;
		}
		vector<float>cenH;
		for (int i = 0; i < l - 1 + Hd + 1; i++)
		{
			float tmp = floor(wl2 + b*i) - wl2;
			cenH.push_back(tmp);
		}
		
		vector<float>regions;
		for (auto i : cenH){
			for (auto j : cenW){
				regions.push_back(j);
				regions.push_back(i);
				regions.push_back(wl);
				regions.push_back(wl);
				regions_xywh.push_back(regions);
				regions.clear();
			}
		}
	}
	for (int i = 0; i < regions_xywh.size(); i++){
		for (int j = 0; j < 4; j++){
			regions_xywh[i][j] = int(round(regions_xywh[i][j]));
		}
		if (regions_xywh[i][0] + regions_xywh[i][2] > W){
			regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W);
		}
		if (regions_xywh[i][1] + regions_xywh[i][3] > H){
			regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H);
		}
	}
	return regions_xywh;

}
static vector<float>get_rmac_features(cv::Mat img, cv::Mat R, shared_ptr<caffe::Net<float>>net){
	vector<float>features;
	vector<cv::Mat> bgr;
	
	cv::split(img, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1.f, -103.939f);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f, -116.779f);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f, -123.68f);
	
	shared_ptr<Blob<float>>input = net->blob_by_name("data");
	const int bias = input->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	
	input->Reshape(1,3,img.size[0],img.size[1]);
	
	memcpy(input->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(input->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(input->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	shared_ptr<Blob<float>>rois = net->blob_by_name("rois");
	
	const int bias1 = rois->offset(0, 1, 0, 0);
	const int bytes1 = bias1*sizeof(float);
	memcpy(rois->mutable_cpu_data(), R.data, bytes1);
	
	net->Forward();
	
	shared_ptr<Blob<float> > feature = net->blob_by_name("rmac/normalized");
	const int kFeatureSize = feature->channels();
	features.resize(kFeatureSize);
	for (int i = 0; i < kFeatureSize; i++) {
		features[i] = feature->data_at(0, i, 0, 0);
	}
	return features;
	
}
static cv::Mat pack_regions_for_network(vector<vector<float>>all_regions){
	int n_regs = all_regions.size();
	Mat R = Mat::zeros(n_regs, 5, CV_32F);
	int cnt = 0;
	for (int i = 0; i < n_regs; i++){
		R.at<float>(i, 0) = 0;
		for (int j = 1; j < 5; j++){
			R.at<float>(i, j) = all_regions[i][j - 1];
		}
	}
	cnt += n_regs;
	R.col(3) = R.col(1) + R.col(3) - 1;
	R.col(4) = R.col(2) + R.col(4) - 1;
	return R;
}

int main(){
	
	shared_ptr<caffe::Net<float>> net;
	
	vector<string> filenames = fs::ListDir(imgdir, { "jpg", "jpeg", "png" });
	
	net.reset(new caffe::Net<float>("../model/deploy_resnet101_my.prototxt", caffe::TEST));
	net->CopyTrainedLayersFrom("../model/model.caffemodel");
	
	for (int i = 0; i < filenames.size(); i++){
		string filepath = imgdir + "/" + filenames[i];
		Mat img = img_resize(filepath);
		
		vector<vector<float>> all_regions = get_rmac_region_coordinates(img);
		
		Mat R = pack_regions_for_network(all_regions);
		cout << "3" << endl;
		vector<float>feature_query = get_rmac_features(img, R, net);
		
	}
	return 0;
}