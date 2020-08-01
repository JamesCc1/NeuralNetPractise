#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include "batch.h"

using namespace std;
using namespace Mini;

class MNIST_LOAD {
private:
	char* filename;
	vector<vector<float>> array;
	vector<int> labels;
	int index;

public:
	MNIST_LOAD(char* path) :filename(path), index(0) {}
	MNIST_LOAD(string path): index(0){
		filename = new char[100];
		path.copy(filename, path.size(), 0);
		filename[path.size()] = '\0';
	}

	void load_data() {
		ifstream file;
		file.open(filename, ios::in);
		assert(file.is_open());
		string line;

		while (getline(file, line)) {
		    auto begIdx = line.find_first_of(',', 0);
		    auto label = atoi(line.substr(0, begIdx).c_str());
		    labels.push_back(label);
		    vector<float> data;
		    auto endIdx = ++begIdx;
		    while (endIdx != string::npos) {
		        endIdx = line.find_first_of(',', begIdx);
		        float x = atof(line.substr(begIdx, endIdx - begIdx).c_str());
			data.push_back(x /255.0);
			begIdx = endIdx + 1;
		    }
		    array.push_back(data);
		}

	}

        inline bool atEnd(){
            bool flag = false;
            if(index >= labels.size()){
                flag = true;
                index = 0;
            }
            return flag;
        }

	pair<Batch<float> , Batch<float>> batch_load(int batch_size) {
	    Batch<float> ys;
	    Batch<float> xs;
	    //cout << "wait" << endl;
	    int end = (batch_size + index) < labels.size() ? (batch_size + index) : labels.size();
	    for (; index < end; index++) {
                Tensor<float> tmp_x(array[index], {1, 784});
		xs.push(std::move(tmp_x));
                vector<float> cc;
                cc.push_back(1.0*labels[index]);
                Tensor<float> tmp_y(cc, {1});
		ys.push(std::move(tmp_y));
	    }
	    return std::move(make_pair(xs, ys));
	}
};
