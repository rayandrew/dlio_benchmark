#include <fcntl.h>
#include <stdio.h>

#include <sys/fcntl.h> 
#include <sys/stat.h>
#include <sys/ioctl.h> 

#include <lustre/lustreapi.h> 

int main(void) {
  // int fd = open("/eagle/MDClimSim/rayandrew/data/striped/train/1979_0000.h5", O_RDONLY);
  // int fd = open("/eagle/MDClimSim/rayandrew/test-data", O_RDONLY);
  // int fd = open("/eagle/MDClimSim/rayandrew/test-data/test", O_RDONLY);
  int fd = open("/eagle/MDClimSim/rayandrew/test-data/test2", O_RDONLY);
  if (fd < 0) {
    perror("open");
    return 1;
  }

  struct lov_user_md opts = { 0 };
  /* Setup Lustre IOCTL striping pattern structure */
  opts.lmm_magic = LOV_USER_MAGIC;
  // opts.lmm_stripe_size = o->lustre_stripe_size;
  // opts.lmm_stripe_offset = o->lustre_start_ost;
  // opts.lmm_stripe_count = o->lustre_stripe_count; 
  
  if (ioctl(fd, LL_IOC_LOV_GETSTRIPE, &opts)) {
    perror("ioctl");
    return 1;
  }

  printf("LMM Stripe size: %ld\n", opts.lmm_stripe_size);
  printf("LMM Stripe offset: %ld\n", opts.lmm_stripe_offset);
  printf("LMM Stripe count: %ld\n", opts.lmm_stripe_count);
  
  return 0;
}
