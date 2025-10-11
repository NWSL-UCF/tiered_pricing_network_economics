# Cloud Data Transfer Pricing Reference

This document provides pricing information for data egress (outbound transfers) from major cloud providers, relevant for network economics analysis and cost modeling.

## Unit Conversion

**Important**: Cloud providers price in **Terabytes (TB)**, not Terabits (Tb):
- 1 Terabyte (TB) = 8 Terabits (Tb)
- 1 Terabit (Tb) = 0.125 Terabytes (TB)

## Major Cloud Provider Pricing

### Amazon Web Services (AWS)

Data transfer OUT to the internet from all AWS regions (as of 2024):

| Volume (per month) | Price per GB | Price per TB | Price per Tb |
|-------------------|--------------|--------------|--------------|
| First 10 TB | $0.090 | $90.00 | $11.25 |
| Next 40 TB | $0.085 | $85.00 | $10.63 |
| Next 100 TB | $0.070 | $70.00 | $8.75 |
| Over 150 TB | $0.050 | $50.00 | $6.25 |

**Free tier**: First 100 GB/month free across all services

**Reference**: [AWS Data Transfer Pricing](https://aws.amazon.com/ec2/pricing/on-demand/#Data_Transfer)

### Microsoft Azure

Data transfer OUT from Azure regions (as of 2024):

| Volume (per month) | Price per GB | Price per TB | Price per Tb |
|-------------------|--------------|--------------|--------------|
| First 10 TB | $0.087 | $87.00 | $10.88 |
| Next 40 TB | $0.083 | $83.00 | $10.38 |
| Next 100 TB | $0.081 | $81.00 | $10.13 |
| Next 350 TB | $0.080 | $80.00 | $10.00 |
| Over 500 TB | $0.076 | $76.00 | $9.50 |

**Free tier**: First 100 GB/month free

**Reference**: [Azure Bandwidth Pricing](https://azure.microsoft.com/en-us/pricing/details/bandwidth/)

### Google Cloud Platform (GCP)

Data egress from GCP to internet (as of 2024):

| Volume (per month) | Price per GB | Price per TB | Price per Tb |
|-------------------|--------------|--------------|--------------|
| 0-1 TB | $0.120 | $120.00 | $15.00 |
| 1-10 TB | $0.110 | $110.00 | $13.75 |
| 10+ TB | $0.080 | $80.00 | $10.00 |

**Free tier**: First 200 GB/month free (with conditions)

**Reference**: [Google Cloud Network Pricing](https://cloud.google.com/vpc/network-pricing)

## Regional Variations

Egress pricing varies by region. Premium rates shown above are for US/Europe regions.

**Asia-Pacific regions** typically cost 10-30% more:
- AWS Asia Pacific: ~$0.12/GB (first 10 TB)
- Azure Asia: ~$0.12/GB (first 10 TB)
- GCP Asia: ~$0.14/GB (0-1 TB)

## Key Pricing Principles

### 1. Egress vs Ingress
- **Egress (outbound)**: Charged at rates above
- **Ingress (inbound)**: Typically FREE across all providers
- **Internal transfers**: May be free within same region, minimal cost between regions

### 2. Free Tiers
All major providers offer monthly free allowances:
- AWS: 100 GB/month
- Azure: 100 GB/month  
- GCP: 200 GB/month (combined with other services)

### 3. Volume Discounts
Significant savings at scale:
- 10-50 TB: ~5-15% discount
- 50-150 TB: ~15-30% discount
- 150+ TB: ~40-50% discount

### 4. Reserved Capacity
Enterprise customers can negotiate:
- Custom pricing for committed volumes
- Flat-rate unlimited transfer plans
- Multi-year discounts (20-30% additional savings)

## Cost Analysis for Network Flow Dataset

Based on our processed dataset (3.607 Terabits = 0.451 Terabytes):

| Provider | Cost Calculation | Estimated Egress Cost |
|----------|-----------------|---------------------|
| AWS | 0.451 TB × $90/TB | **$40.59** |
| Azure | 0.451 TB × $87/TB | **$39.24** |
| GCP | 0.451 TB × $120/TB | **$54.12** |

**Note**: These are first-tier prices. Actual costs depend on monthly volumes and regional distribution.

## CDN Alternatives

Content Delivery Networks often offer better rates for high-volume egress:

### Cloudflare
- Bandwidth Alliance: Free egress from supported clouds
- Standard: ~$0.04-0.08/GB ($40-80/TB)
- Enterprise: Custom pricing, often flat-rate

### Fastly
- Pay-as-you-go: ~$0.08-0.12/GB
- Volume discounts available

### AWS CloudFront (CDN)
- Cheaper than direct EC2 egress
- US/Europe: $0.085/GB (first 10 TB)
- Significant savings for public content delivery

## Academic and Research Discounts

Many providers offer reduced/free pricing for research:
- **AWS Research Credits**: Up to $100,000
- **Google Cloud Research Credits**: Up to $5,000
- **Azure for Research**: Custom grants

**Reference**: 
- [AWS Cloud Credits for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/)
- [Google Cloud Research Credits](https://edu.google.com/programs/credits/research/)

## Cost Optimization Strategies

### 1. Geographic Routing
- Route traffic through cheaper regions when possible
- Use CDN edge locations strategically

### 2. Compression
- Enable gzip/brotli compression (50-80% reduction)
- Can reduce effective costs by half

### 3. Caching
- Edge caching reduces origin egress
- Client-side caching minimizes repeated transfers

### 4. Multi-Cloud Strategy
- Distribute traffic across providers
- Leverage free tiers from multiple vendors

### 5. Direct Peering
- Large enterprises can establish direct connections
- Bypass public internet egress entirely
- AWS Direct Connect, Azure ExpressRoute, GCP Interconnect

## Real-World Cost Implications

For a typical content provider serving 1 PB/month:

| Scenario | Monthly Cost | Annual Cost |
|----------|--------------|-------------|
| Single cloud (AWS) | $65,000 | $780,000 |
| With CDN optimization | $40,000 | $480,000 |
| Multi-cloud + CDN | $30,000 | $360,000 |
| Enterprise negotiated | $20,000 | $240,000 |

**Potential savings: 60-70% with optimization**

## Pricing Trends

Historical trends (2020-2024):
- Average prices decreased 15-20%
- Competition from specialized CDNs
- Introduction of free tiers and credits
- Volume discounts became more aggressive

Expected future trends:
- Continued price pressure from competition
- More generous free tiers
- Flat-rate pricing for large customers
- Edge computing may change pricing models

## References and Resources

### Official Pricing Pages
1. [AWS EC2 Data Transfer Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
2. [Azure Bandwidth Pricing](https://azure.microsoft.com/en-us/pricing/details/bandwidth/)
3. [Google Cloud Network Pricing](https://cloud.google.com/vpc/network-pricing)
4. [Cloudflare Bandwidth Pricing](https://www.cloudflare.com/plans/)

### Pricing Calculators
- [AWS Pricing Calculator](https://calculator.aws/)
- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)

### Academic Resources
- Barroso, L. A., et al. (2019). "The Datacenter as a Computer: Designing Warehouse-Scale Machines"
- Greenberg, A., et al. (2008). "The Cost of a Cloud: Research Problems in Data Center Networks"
- [Internet2 Network Pricing](https://internet2.edu/services/network/) - Academic network pricing models

### Industry Reports
- Gartner Cloud Infrastructure and Platform Services Market Guide (Updated annually)
- Forrester Cloud Infrastructure Market Research
- 451 Research Cloud Price Index

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Maintained by**: Network Economics Research Team

**Note**: Prices are subject to change. Always verify current rates on official provider websites before cost modeling.

