#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
//#include "Eigen\core"
#include "narivectorpp.h"
#include "info.h"
#include <Eigen/Dense>
#include <sys/stat.h>
#include "direct.h"
#include <naricommon.h>
#include <nariinfocontroller.h>
#include <narifile.h>

template< class T >
void write_vector(std::vector<T> &v, const std::string filename) {
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "wb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	fwrite(v.data(), sizeof(T), v.size(), fp);
	fclose(fp);
}

long get_file_size(std::string filename)
{
	FILE *fp;
	struct stat st;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	fstat(_fileno(fp), &st);
	fclose(fp);
	return st.st_size;
}

template< class T >
void read_vector(std::vector<T> &v, const std::string filename) {

	auto num = get_file_size(filename) / sizeof(T);
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	v.resize(num);
	fread(v.data(), sizeof(T), num, fp);
	fclose(fp);
}

template<typename T>
void write_matrix_raw_and_txt(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data, std::string filename)
{
	//////////////////////////////////////////////////////////////
	// W�̏����o��												//
	// row���B��w�̐��Ccol�����w�̐�							//			
	// �d�݂̉������s���ꍇ�́C�e�s��؂�o����reshape���s��  //
	//////////////////////////////////////////////////////////////
	size_t rows = data.rows();
	size_t cols = data.cols();
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Data;
	Data = data;
	std::ofstream fs1(filename + ".txt");
	fs1 << "rows = " << rows << std::endl;
	fs1 << "cols = " << cols << std::endl;
	fs1 << typeid(Data).name() << std::endl;
	fs1.close();
	std::vector<T> save_data(rows * cols);
	Data.resize(rows * cols, 1);
	for (size_t i = 0; i < save_data.size(); i++)
		save_data[i] = Data(i, 0);
	write_vector(save_data, filename + ".raw");
	Data.resize(rows, cols);
}


void main(int argc, char *argv[]) {
	info input_info;
	input_info.input(argv[1]);

	//�e�L�X�g�f�[�^�ǂݍ���	
	std::vector<std::string> fcase;
	std::vector<std::string> rcase;
	std::ifstream f_case(input_info.dir_list + input_info.case_flist);
	std::ifstream r_case(input_info.dir_list + input_info.case_rlist);
	std::string buf_ft;
	std::string buf_rt;
	while (f_case&& getline(f_case, buf_ft))
	{
		fcase.push_back(buf_ft);
	}
	while (r_case&& getline(r_case, buf_rt))
	{
		rcase.push_back(buf_rt);
	}
	//�w�K�S�f�[�^��,���ꂼ��̎�����
	int n = fcase.size() - 1;
	int Fd = input_info.fd;
	int Rd = input_info.rd;
	std::vector<double> L2error;
	//Leave-one-out ���[�v
	for (int i = 0; i < fcase.size(); i++) {
		//�t�@�C���ǂݍ���
		//�����O�`��LS�听���X�R�A
		std::vector<double> Fl;
		read_vector(Fl, input_info.dir_score + "Fl/" + fcase[i] + "/mat.raw");
		// ������`��LS�听���X�R�A(�������܂�)
		std::vector<double> Ref;
		read_vector(Ref, input_info.dir_score + "Ref/" + rcase[i] + "/mat.raw");
		//�e���̕��U��ǂݍ���
		std::vector<double> r_cov;
		std::ifstream covtxt(input_info.dir_score + "Ref/" + rcase[i] + "/eval.txt");
		std::string buf_co;
		while (covtxt&& getline(covtxt, buf_co))
		{
			r_cov.push_back(stod(buf_co));
		}
		//���ꂼ��w�K,�e�X�g�f�[�^�̃X�R�A�̂ݔ����o��
		std::vector<double> Fl_te;    //�e�X�g����
		std::vector<double> Ref_te;   //�e�X�g����
		for (int k = 0; k < fcase.size(); k++) {
			for (int l = 0; l < Fd; l++) {
				//�f�o�b�O�̎��͂�����ύX����ׂ�
				//int s = j*Fd + k;
				int s = k*(fcase.size() - 2) + l;
				if ((k == i) && (l < Fd)) {
					Fl_te.push_back(Fl[s]);
				}
			}
		}
		for (int k = 0; k < fcase.size(); k++) {
			for (int l = 0; l < Rd; l++) {
				//�f�o�b�O�̎��͂�����ύX����ׂ�
				//int s = j*Rd + k;
				int s = k*(fcase.size() - 2) + l;
				if ((k == i) && (l < Rd)) {
					Ref_te.push_back(Ref[s]);
				}
			}
		}
		Eigen::MatrixXd Xt_0 = Eigen::Map<Eigen::MatrixXd>(&Fl_te[0], 1, Fd);
		Eigen::MatrixXd Xt = Xt.Ones(1, Fd + 1);
		Xt.block(0, 1, 1, Fd) = Xt_0;
		//�W�����i�[���镶������`
		Eigen::MatrixXd co_sum = co_sum.Zero(Fd + 1, Rd);

		//�w�KL�@���[�[�[�v
		for (int j = 0; j < fcase.size(); j++) {
			if (i == j) continue;
			std::vector<double> Fl_tr;
			std::vector<double> Ref_tr;
			for (int k = 0; k < fcase.size(); k++) {
				for (int l = 0; l < Fd; l++) {
					//�f�o�b�O�̎��͂�����ύX����ׂ�
					//int s = j*Fd + k;
					int s = k*(fcase.size() - 2) + l;
					if ((k == i) || (k == j)) continue;
					else if (l < Fd) {
						Fl_tr.push_back(Fl[s]);
					}
				}
			}
			for (int k = 0; k < fcase.size(); k++) {
				for (int l = 0; l < Rd; l++) {
					//�f�o�b�O�̎��͂�����ύX����ׂ�
					//int s = j*Rd + k;
					int s = k*(fcase.size() - 2) + l;
					if ((k == i) || (k == j)) continue;
					else if (l < Rd) {
						Ref_tr.push_back(Ref[s]);
					}
				}
			}
			Eigen::MatrixXd X_0 = Eigen::Map<Eigen::MatrixXd>(&Fl_tr[0], Fd, n - 1);
			Eigen::MatrixXd X = X.Ones(n - 1, Fd + 1);
			X.block(0, 1, n-1, Fd) = X_0.transpose();
			Eigen::MatrixXd Y__ = Eigen::Map<Eigen::MatrixXd>(&Ref_tr[0], Rd, n - 1);
			Eigen::MatrixXd Y = Y__.transpose();
			Eigen::MatrixXd linear_0 = X.transpose()*X;
			Eigen::MatrixXd linear = linear_0.inverse()*X.transpose()*Y; //�W���Z�o
			co_sum += linear;
		}
		//�w�K�f�[�^L�@�ŏo�����W���̕��ς��Ƃ�
		Eigen::MatrixXd co_mean = co_sum / (n - 1);
		Eigen::MatrixXd linear_result = Xt*co_mean;
		std::cout << linear_result << std::endl;
		std::stringstream dirOUT;
		dirOUT << input_info.dir_out << fcase[i] << "/linear";
		write_matrix_raw_and_txt(linear_result, dirOUT.str());
		std::ofstream mat_result(dirOUT.str() + ".txt");
		double sum_E = 0;
		for (int j = 0; j < Rd; j++) {
			mat_result << linear_result(0, j) << std::endl;
			double dev = sqrt(r_cov[j]);
			double reg_l = linear_result(0, j) / dev; //���K����\���X�R�A
			double reg_a = Ref_te[j] / dev; //���K���㐳���X�R�A
			sum_E += (reg_l - reg_a)*(reg_l - reg_a);
		}
		L2error.push_back(sqrt(sum_E));
	}
	std::stringstream dirOUT2;
	dirOUT2 << input_info.dir_out << "L2error";
	std::ofstream mat_result2(dirOUT2.str() + "_LR.txt");
	for (int i = 0; i < fcase.size(); i++) {
		mat_result2 << L2error[i] << std::endl;
	}
}
