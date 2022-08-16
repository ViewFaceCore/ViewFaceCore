using FreeSql;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore.Demo.VideoForm.Models;

namespace ViewFaceCore.Demo.VideoForm
{
    public class DefaultDbContext : DbContext
    {
        public DbSet<UserInfo> UserInfo { get; set; }

        protected override void OnConfiguring(DbContextOptionsBuilder builder)
        {
            builder.UseFreeSql(ConnectionManager.Instance.GetDefault());
        }
    }

    class ConnectionManager
    {
        private string ConnectionString
        {
            get
            {
                return "Data Source=./data.db; Pooling=true;Min Pool Size=5";
            }
        }

        private readonly IFreeSql dbHandler = null;

        public static ConnectionManager Instance = new ConnectionManager();

        private ConnectionManager()
        {
            dbHandler = new FreeSql.FreeSqlBuilder()
                .UseConnectionString(FreeSql.DataType.Sqlite, ConnectionString)
                .UseAutoSyncStructure(true) //自动同步实体结构【开发环境必备】，FreeSql不会扫描程序集，只有CRUD时才会生成表。
                .UseMonitorCommand(cmd => Console.Write(cmd.CommandText))
                .Build(); //请务必定义成 Singleton 单例模式
        }

        public IFreeSql GetDefault()
        {
            return dbHandler;
        }
    }
}
