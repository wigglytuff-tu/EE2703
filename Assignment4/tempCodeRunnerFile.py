def coeff_plotter(coeffs,fname):
    plt.grid()
    plt.semilogy(abs(coeffs),'o',color = 'r',markersize = 4)
    plt.title('Fourier Coefficients for {} by direct integration (semilog)'.format(fname))
    plt.xlabel('n')
    plt.ylabel('Fourier Coefficients') 
    plt.show()


    plt.grid()
    plt.loglog(abs(coeffs),'o',color = 'r',markersize = 4)
    plt.title('Fourier Coefficients for {} by direct integration (loglog)'.format(fname))
    plt.xlabel('n')
    plt.ylabel('Fourier Coefficients') 
    plt.show()

coeff_plotter(a,'e^x')
coeff_plotter(b,'cos(cos(x))')