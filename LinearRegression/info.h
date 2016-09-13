#pragma once
#ifndef __INFO__
#define __INFO__
#include<iostream>
#include <string>
#include <algorithm>
#include <vector>

#include <naricommon.h>
#include <nariinfocontroller.h>
#include <narifile.h>

struct info
{
	std::string dir_score;
	std::string dir_out;
	std::string dir_list;
	std::string case_flist;
	std::string case_rlist;
	int fd;
	int rd;

	inline void input(const std::string &path)
	{
		nari::infocontroller info;
		info.load(path);
		dir_score = nari::file::add_delim(info.get_as_str("dir_score"));
		dir_out = nari::file::add_delim(info.get_as_str("dir_out"));
		dir_list = nari::file::add_delim(info.get_as_str("dir_txt"));
		case_flist = info.get_as_str("case_f");
		case_rlist = info.get_as_str("case_r");

		fd = info.get_as_int("Fl_d"); //�����ϐ��̐�
		rd = info.get_as_int("Ref_d"); //�ړI�ϐ��̐�
	}

};
#endif