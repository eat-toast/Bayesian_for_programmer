from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pymc as pm

matplotlib.rc('font', family='Malgun Gothic')

figsize(12.5, 4)

count_data = np.loadtxt("책 자료\\Ch01\\txtdata.csv")

n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("시간(일수)",fontsize=13)
plt.ylabel("수신한 문자 메시지 개수",fontsize=13)
# plt.title("사용자의 메시징 습관이 시간에 따라 변하는가?")
plt.xlim(0, n_count_data)


alpha = 1.0 / count_data.mean() # count_data 변수는 문자 메시지 개수를 저장한다.

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential('lambda_2', alpha)

tau = pm.DiscreteUniform("tau", lower = 0, upper = n_count_data)

print('Random ouput: ', tau.random(), tau.random(), tau.random())

@pm.deterministic
def lambda_(tau=tau, lambda_1 = lambda_1, lambda_2= lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1 # lambda_1은 tau 이전 lambda 이다
    out[tau:] = lambda_2 # lambda_2은 tau 이후 lambda 이다
    return out


observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)

model = pm.Model([observation, lambda_1, lambda_2, tau])

# 3장에서 이 코드를 더 자세히 설명할 예정
# 30,000 = (40,000 - 10,000
mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

figsize(14.5, 10)
# 표본의 히스토그램:

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="$\lambda_1$의 사후확률분포", color="#A60628", normed=True)
plt.legend(loc="upper left")
#plt.title(r"""모수 $\lambda_1,\;\lambda_2,\;\tau$의 사후확률분포""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ 값")
plt.ylabel("밀도", fontsize=13)
ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="$\lambda_2$의 사후확률분포", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ 값")
plt.ylabel("밀도",fontsize=13)

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"$\tau$의 사후확률분포",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel(r"$\tau$ (일수)",fontsize=13)
plt.ylabel("확률",fontsize=13)

