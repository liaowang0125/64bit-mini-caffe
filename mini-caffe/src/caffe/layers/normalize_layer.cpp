
#include "caffe/layers/normalize_layer.hpp"
#include <caffe/util/math_functions.hpp>
#include <vector>
#include <math.h>
#include <iostream>
using namespace std;
namespace caffe{
	template <typename Dtype>
	void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		num_images_ = bottom[0]->num();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		num_channels_ = bottom[0]->channels();
		eps = 1e-8;
		
	}
	template <typename Dtype>
	void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
	}
	template <typename Dtype>
	void NormalizeLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		Dtype * top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const int count1 = bottom[0]->count();
		const int count2 = top[0]->count();
		vector<float>tmp;
		
		for (int i = 0; i < bottom[0]->num(); i++){
			float sum = 0;
			for (int j = 0; j < bottom[0]->channels(); j++){
				int idx = i*bottom[0]->channels() + j;
				sum += bottom_data[idx] * bottom_data[idx];
				if (j == bottom[0]->channels() - 1){
					sum = sqrt(sum);
					sum += eps;
					tmp.push_back(sum);
				}
			}
		}
		
		for (int i = 0; i < bottom[0]->num(); i++){
			for (int j = 0; j < bottom[0]->channels(); j++){
				int idx = i*bottom[0]->channels() + j;
				top_data[idx] = bottom_data[idx] / tmp[i];
			}
		}
		tmp.clear();
		/*for (int i = 0; i < count2; i++){
			cout << top_data[count2] << "  ";
		}*/

	}

	INSTANTIATE_CLASS(NormalizeLayer);
	REGISTER_LAYER_CLASS(Normalize);
}