#pragma once

/// <summary>
/// 释放
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="ptr"></param>
template <typename T>
inline void _dispose(T &ptr)
{
	if (ptr != nullptr)
	{
		try
		{
			delete ptr;
			ptr = nullptr;
		}
		catch (int e)
		{
		}
	}
}

