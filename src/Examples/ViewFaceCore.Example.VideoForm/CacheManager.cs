using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Caching;
using ViewFaceCore.Demo.VideoForm.Models;
using ViewFaceCore.Core;

namespace ViewFaceCore.Demo.VideoForm
{
    internal class CacheManager
    {
        private MemoryCache cache = MemoryCache.Default;
        private const string key = "DefaultViewUserKey";

        public static CacheManager Instance = new CacheManager();

        private CacheManager()
        {

        }

        public List<UserInfo> Get()
        {
            object data = cache.Get(key);
            if (data != null && data is List<UserInfo>)
            {
                return data as List<UserInfo>;
            }
            return null;
        }

        public UserInfo Get(FaceRecognizer faceRecognizer, float[] data)
        {
            if (faceRecognizer == null || data == null)
            {
                return null;
            }
            List<UserInfo> users = Get();
            if (users == null || !users.Any())
            {
                return null;
            }
            foreach (var item in users)
            {
                float[] userExtractData = item.GetExtractData();
                if (userExtractData.Length > data.Length)
                {
                    userExtractData = new float[data.Length];
                    Array.Copy(data, userExtractData, data.Length);
                }
                if (faceRecognizer.IsSelf(data, userExtractData))
                {
                    return item;
                }
            }
            return null;
        }

        public void Refesh()
        {
            using (DefaultDbContext db = new DefaultDbContext())
            {
                var allUsers = db.UserInfo.Where(p => !p.IsDelete).ToList(p => new UserInfo()
                {
                    Id = p.Id,
                    Name = p.Name,
                    Age = p.Age,
                    Gender = p.Gender,
                    Phone = p.Phone,
                    Extract = p.Extract,
                });
                if (allUsers != null)
                {
                    cache.Set(key, allUsers, new DateTimeOffset(DateTime.Now.AddDays(3650)));
                }
            }
        }
    }
}
