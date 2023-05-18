# Bui Quang Khai - 20001558
using DelimitedFiles
using Plots
function readData(path, numFeatures=11)
    ogdata = readdlm(path, ';')
    data = ogdata[2:end, :]
    ŷ = data[:, end]
    X = Float64.([ones(length(ŷ)) data[:, 1:numFeatures]])
    return X, ŷ
end
Xwhite, ywhite = readData("C:/Users/Asus/OneDrive/Desktop/data/winequality-white.txt")
Xred, yred = readData("C:/Users/Asus/OneDrive/Desktop/hus/thirdyear 2/MachineLearing/data/winequality-red.txt")

#train
train(X, y) = inv(X' * X) * X' * y

θwhite = train(Xwhite, ywhite)
θred = train(Xred, yred)

#predict
predict(X, θ) = θ * X

predwhite = predict(θwhite, Xred)
predred = predict(θred, Xwhite)

function accuracy(predData, y, ϵ=0.5)
    n = length(y)
    diff = y .- predData
    count = sum(abs.(diff) .< ϵ)
    return count / n
end

println(accuracy(predred, ywhite))

println(accuracy(predwhite, yred))

# output:
# 0.5202123315639037
# 0.3420888055034397

