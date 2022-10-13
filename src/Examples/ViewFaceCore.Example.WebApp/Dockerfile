#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:6.0-bullseye-slim AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["Examples/ViewFaceCore.Example.WebApp/ViewFaceCore.Example.WebApp.csproj", "Examples/ViewFaceCore.Example.WebApp/"]
RUN dotnet restore "Examples/ViewFaceCore.Example.WebApp/ViewFaceCore.Example.WebApp.csproj"
COPY . .
WORKDIR "/src/Examples/ViewFaceCore.Example.WebApp"
RUN dotnet build "ViewFaceCore.Example.WebApp.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "ViewFaceCore.Example.WebApp.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN sed -i 's|security.debian.org/debian-security|mirrors.ustc.edu.cn/debian-security|g' /etc/apt/sources.list
RUN apt update && apt install apt-utils libgdiplus libc6-dev libgomp1 -y && apt clean

ENTRYPOINT ["dotnet", "ViewFaceCore.Example.WebApp.dll"]