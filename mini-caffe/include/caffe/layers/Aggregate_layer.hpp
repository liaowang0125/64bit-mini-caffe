#ifndef CAFFE_AGGREGATE_LAYER_HPP_
#define CAFFE_AGGREGATE_LAYER_HPP_
#include "caffe/layer.hpp"
namespace caffe {
	template <typename Dtype>
	class AggregateLayer : public Layer<Dtype> {
	public:
		explicit AggregateLayer(const LayerParameter& param)
			:Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {}
		float eps;
		int height_;
		int width_;
		int num_images_;
		int num_channels_;
	};
}

#endif