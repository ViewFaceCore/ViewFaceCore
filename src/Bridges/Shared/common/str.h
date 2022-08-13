/** Copyright 2019 Mish7913 <Mish7913@gmail.com> **/

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 *
 */

#include <algorithm>
#include <iostream>
#include <string.h>
#include <string>
#include <map>

#ifndef STR_7913_H
#define STR_7913_H

namespace str
{
    std::map<int, std::string> split_to_map(std::string str, std::string sep);
    std::wstring str_to_wstr(const std::string &s);
    std::string wstr_to_str(const std::wstring &ws);
    std::string substr(std::string str, int pos, int len);
    std::string delete_html_tags(std::string str, int mode);
    std::string replace(std::string str, int pos, int len, std::string rep);
    std::string find_replace(std::string str, std::string find_str, std::string rep);
    std::string int_to_str(int num);
    int find(std::string str, std::string find_str, int pos);
    int length(std::string str);
    int str_to_int(std::string str);
}
#endif
